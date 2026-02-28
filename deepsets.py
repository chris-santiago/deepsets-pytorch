
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def _masked_pool(
    tensor: torch.Tensor,
    pool_type: str,
    mask: Optional[torch.Tensor] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Pool a (B, M, D) tensor along the set dimension (dim=1).

    Correctly handles masks for all pool types:
      - sum/mean: zero out padding before summing.
      - max: fill padding with -inf so masked positions never win,
             even when all active values are negative.

    Args:
        tensor:   (B, M, D)
        pool_type: 'sum', 'max', or 'mean'
        mask:     (B, M) float/bool — 1 = active element, 0 = padding
        keepdim:  if True, retain the M dimension as size 1 in output
    Returns:
        (B, [1,] D) depending on keepdim
    """
    if pool_type == 'sum':
        if mask is not None:
            tensor = tensor * mask.unsqueeze(-1)
        return tensor.sum(dim=1, keepdim=keepdim)

    elif pool_type == 'max':
        if mask is not None:
            tensor = tensor.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        return tensor.max(dim=1, keepdim=keepdim)[0]

    elif pool_type == 'mean':
        if mask is not None:
            masked = tensor * mask.unsqueeze(-1)
            count = mask.sum(dim=1, keepdim=True)   # (B, 1)
            if keepdim:
                count = count.unsqueeze(-1)          # (B, 1, 1)
            return masked.sum(dim=1, keepdim=keepdim) / count.clamp(min=1)
        return tensor.mean(dim=1, keepdim=keepdim)

    else:
        raise ValueError(
            f"Unknown pool_type '{pool_type}'. Expected 'sum', 'max', or 'mean'."
        )


class DeepSetsInvariant(nn.Module):
    """
    Deep Sets architecture for permutation invariant functions.

    Implements Theorem 2:  f(X) = ρ(Σ_{x ∈ X} φ(x))

    φ transforms each element independently; the results are pooled into a
    single vector and passed through ρ to produce the final output.

    Args:
        input_dim:       Dimension of each set element.
        phi_hidden_dims: Hidden layer widths for φ (element-wise network).
        rho_hidden_dims: Hidden layer widths for ρ (aggregation network).
        output_dim:      Output dimension.
        pool_type:       Aggregation: 'sum', 'max', or 'mean'.
        dropout:         Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        input_dim: int,
        phi_hidden_dims: List[int],
        rho_hidden_dims: List[int],
        output_dim: int,
        pool_type: str = 'sum',
        dropout: float = 0.0,
    ):
        super().__init__()
        if pool_type not in ('sum', 'max', 'mean'):
            raise ValueError(f"Unknown pool_type '{pool_type}'. Expected 'sum', 'max', or 'mean'.")
        self.pool_type = pool_type

        # φ network: transforms each element independently
        phi_layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in phi_hidden_dims:
            phi_layers.append(nn.Linear(prev_dim, hidden_dim))
            phi_layers.append(nn.ReLU())
            if dropout > 0:
                phi_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)

        # ρ network: processes the pooled representation
        rho_layers: List[nn.Module] = []
        prev_dim = phi_hidden_dims[-1] if phi_hidden_dims else input_dim
        for hidden_dim in rho_hidden_dims:
            rho_layers.append(nn.Linear(prev_dim, hidden_dim))
            rho_layers.append(nn.ReLU())
            if dropout > 0:
                rho_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, M, input_dim)
            mask: (B, M) — 1 for active elements, 0 for padding
        Returns:
            (B, output_dim)
        """
        phi_out = self.phi(x)                                    # (B, M, phi_dim)
        pooled  = _masked_pool(phi_out, self.pool_type, mask)    # (B, phi_dim)
        return self.rho(pooled)                                  # (B, output_dim)


class PermutationEquivariantLayer(nn.Module):
    """
    Single permutation equivariant layer (Lemma 3).

    Implements:  f(x) = Λx + Γ · pool(X) · 1

    where Λ and Γ are learnable linear maps. The pooled global context is
    broadcast back to every element, making the transformation equivariant:
    permuting the input permutes the output identically.

    Args:
        input_dim:  Input feature dimension per element.
        output_dim: Output feature dimension per element.
        pool_type:  Pooling for global context: 'max' or 'sum'.
    """

    def __init__(self, input_dim: int, output_dim: int, pool_type: str = 'max'):
        super().__init__()
        if pool_type not in ('sum', 'max'):
            raise ValueError(
                f"PermutationEquivariantLayer pool_type must be 'sum' or 'max', "
                f"got '{pool_type}'."
            )
        self.pool_type  = pool_type
        self.lambda_net = nn.Linear(input_dim, output_dim)  # Λ: per-element transform
        self.gamma_net  = nn.Linear(input_dim, output_dim)  # Γ: global-context transform

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, M, input_dim)
            mask: (B, M)
        Returns:
            (B, M, output_dim)
        """
        lambda_out = self.lambda_net(x)                                       # (B, M, D_out)
        pooled     = _masked_pool(x, self.pool_type, mask, keepdim=True)      # (B, 1,  D_in)
        gamma_out  = self.gamma_net(pooled)                                   # (B, 1,  D_out)
        return lambda_out + gamma_out                                         # broadcast over M


class DeepSetsEquivariant(nn.Module):
    """
    Deep Sets architecture for permutation equivariant functions.

    Stacks PermutationEquivariantLayer blocks, optionally followed by a
    final pooling step to produce an invariant output.

    Args:
        input_dim:  Dimension of each set element.
        hidden_dims: Hidden widths for each equivariant layer.
        output_dim: Output dimension per element (or global if final_pool set).
        pool_type:  Pooling inside each equivariant layer: 'sum' or 'max'.
        final_pool: If set ('sum', 'max', 'mean'), pool across the set at the
                    end to produce an invariant (B, output_dim) output.
                    If None, output is per-element (B, M, output_dim).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        pool_type: str = 'max',
        final_pool: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if final_pool is not None and final_pool not in ('sum', 'max', 'mean'):
            raise ValueError(f"Unknown final_pool '{final_pool}'.")
        self.final_pool = final_pool

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(PermutationEquivariantLayer(prev_dim, hidden_dim, pool_type))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(PermutationEquivariantLayer(prev_dim, output_dim, pool_type))

        # ModuleList registers all submodules for parameter tracking.
        # We use manual iteration (not nn.Sequential's forward) so we can
        # pass `mask` selectively to PermutationEquivariantLayer instances.
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, M, input_dim)
            mask: (B, M)
        Returns:
            (B, M, output_dim)  if final_pool is None
            (B, output_dim)     if final_pool is set
        """
        out = x
        for layer in self.layers:
            if isinstance(layer, PermutationEquivariantLayer):
                out = layer(out, mask)
            else:
                out = layer(out)

        if self.final_pool is not None:
            out = _masked_pool(out, self.final_pool, mask)

        return out


class DeepSetsConditional(nn.Module):
    """
    Deep Sets conditioned on a context vector z.

    Implements: f(X | z) = ρ(Σ_{x ∈ X} φ(x | z))

    Three strategies for fusing context into φ:
      'concat': concatenate z to each element before φ.
      'film':   Feature-wise Linear Modulation after the first φ layer:
                  h = φ_first(x)
                  h = relu(γ(z) * h + β(z))
                  φ_out = φ_rest(h)
      'add':    add a linear projection of z to each element before φ.

    Args:
        input_dim:       Dimension of each set element.
        context_dim:     Dimension of conditioning vector.
        phi_hidden_dims: Hidden widths for φ (must be non-empty for 'film').
        rho_hidden_dims: Hidden widths for ρ.
        output_dim:      Output dimension.
        pool_type:       Pooling: 'sum', 'max', or 'mean'.
        fusion_type:     Context fusion strategy: 'concat', 'film', or 'add'.
        context_in_rho:  If True (default), concatenate z to the ρ input so ρ
                         has direct access to the conditioning variable.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        phi_hidden_dims: List[int],
        rho_hidden_dims: List[int],
        output_dim: int,
        pool_type: str = 'sum',
        fusion_type: str = 'concat',
        context_in_rho: bool = True,
    ):
        super().__init__()
        if pool_type not in ('sum', 'max', 'mean'):
            raise ValueError(f"Unknown pool_type '{pool_type}'.")
        if fusion_type not in ('concat', 'film', 'add'):
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. Expected 'concat', 'film', or 'add'."
            )
        if fusion_type == 'film' and not phi_hidden_dims:
            raise ValueError("phi_hidden_dims must be non-empty when fusion_type='film'.")

        self.pool_type      = pool_type
        self.fusion_type    = fusion_type
        self.context_in_rho = context_in_rho

        # --- φ network ---
        if fusion_type == 'concat':
            self.phi = self._make_phi(input_dim + context_dim, phi_hidden_dims)

        elif fusion_type == 'film':
            # First linear layer applied before FiLM modulation
            self.phi_first  = nn.Linear(input_dim, phi_hidden_dims[0])
            self.film_gamma = nn.Linear(context_dim, phi_hidden_dims[0])
            self.film_beta  = nn.Linear(context_dim, phi_hidden_dims[0])
            # Remaining layers applied after FiLM + ReLU
            rest: List[nn.Module] = []
            prev = phi_hidden_dims[0]
            for d in phi_hidden_dims[1:]:
                rest.extend([nn.Linear(prev, d), nn.ReLU()])
                prev = d
            self.phi_rest = nn.Sequential(*rest)

        else:  # 'add'
            self.context_proj = nn.Linear(context_dim, input_dim)
            self.phi = self._make_phi(input_dim, phi_hidden_dims)

        if phi_hidden_dims:
            phi_out_dim = phi_hidden_dims[-1]
        elif fusion_type == 'concat':
            phi_out_dim = input_dim + context_dim
        else:
            phi_out_dim = input_dim

        # --- ρ network ---
        rho_in_dim = phi_out_dim + (context_dim if context_in_rho else 0)
        rho_layers: List[nn.Module] = []
        prev_dim = rho_in_dim
        for hidden_dim in rho_hidden_dims:
            rho_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)

    @staticmethod
    def _make_phi(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = input_dim
        for d in hidden_dims:
            layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, M, input_dim)
            context: (B, context_dim)
            mask:    (B, M)
        Returns:
            (B, output_dim)
        """
        B, M, _ = x.shape

        if self.fusion_type == 'concat':
            ctx_exp = context.unsqueeze(1).expand(-1, M, -1)
            phi_out = self.phi(torch.cat([x, ctx_exp], dim=-1))

        elif self.fusion_type == 'film':
            h     = self.phi_first(x)                          # (B, M, D0)
            gamma = self.film_gamma(context).unsqueeze(1)      # (B, 1, D0)
            beta  = self.film_beta(context).unsqueeze(1)       # (B, 1, D0)
            h     = F.relu(gamma * h + beta)                   # FiLM + activation
            phi_out = self.phi_rest(h)                         # remaining layers

        else:  # 'add'
            ctx_proj = self.context_proj(context).unsqueeze(1) # (B, 1, input_dim)
            phi_out  = self.phi(x + ctx_proj)

        pooled = _masked_pool(phi_out, self.pool_type, mask)   # (B, phi_out_dim)

        rho_in = torch.cat([pooled, context], dim=-1) if self.context_in_rho else pooled
        return self.rho(rho_in)
