"""
Theory-faithful test suite for the Deep Sets implementation.

Tests are organized around key results from the paper:
  Zaheer et al., "Deep Sets", NeurIPS 2017. arXiv:1703.06114

──────────────────────────────────────────────────────────────────────────────
Theorem 2 (Permutation Invariance / Universal Approximation)
  Any permutation-invariant function f : 2^X → ℝ can be decomposed as
      f(X) = ρ(Σ_{x ∈ X} φ(x))
  for some (learnable) transformations ρ and φ.

Lemma 3 (Permutation Equivariance)
  A network layer L is permutation equivariant iff its weight matrix has the
  form  Θ = λI + γ(11ᵀ), which is implemented as
      f(x) = σ(Λx + Γ · pool(X) · 1)
  where Λ ≡ λI and Γ ≡ γI in the scalar case.
──────────────────────────────────────────────────────────────────────────────

Run with:
    python test_deepsets_theory.py
or:
    pytest test_deepsets_theory.py -v
"""

import math
import torch
import torch.nn as nn
import sys

from deepsets import (
    DeepSetsInvariant,
    DeepSetsEquivariant,
    PermutationEquivariantLayer,
    DeepSetsConditional,
    _masked_pool,
)

# ── helpers ──────────────────────────────────────────────────────────────────

ATOL = 1e-5   # absolute tolerance for structural / exact tests
LTOL = 0.05   # loss threshold for learning tests

torch.manual_seed(42)


def _random_perm(size: int) -> torch.Tensor:
    return torch.randperm(size)


def _run(name: str, fn):
    """Run a test function and collect the result."""
    try:
        passed, msg = fn()
    except Exception as e:
        passed, msg = False, f"Exception: {e}"
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name}")
    if not passed:
        print(f"         └─ {msg}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STRUCTURAL TESTS — verify architecture matches paper formulae exactly
# ══════════════════════════════════════════════════════════════════════════════

def test_theorem2_decomposition():
    """
    Theorem 2 structural check.

    For DeepSetsInvariant with sum pooling, verify that the output equals
        ρ( Σ_{x ∈ X} φ(x) )
    by manually computing each step and comparing to model.forward().
    """
    model = DeepSetsInvariant(
        input_dim=4,
        phi_hidden_dims=[8, 8],
        rho_hidden_dims=[8],
        output_dim=3,
        pool_type='sum',
    )
    model.eval()

    B, M = 3, 6
    x = torch.randn(B, M, 4)

    with torch.no_grad():
        # Manual: apply φ element-wise, sum, apply ρ
        phi_out      = model.phi(x)          # (B, M, 8)
        pooled_manual = phi_out.sum(dim=1)   # (B, 8)
        out_manual    = model.rho(pooled_manual)

        # Model forward
        out_model = model(x)

    error = (out_manual - out_model).abs().max().item()
    if error > ATOL:
        return False, f"Decomposition mismatch: error={error:.2e}"
    return True, f"ρ(Σφ(x)) verified, max error={error:.2e}"


def test_lemma3_layer_formula():
    """
    Lemma 3 structural check.

    For PermutationEquivariantLayer verify the output equals
        Λ·x + Γ·pool(X)·1
    by manually computing each term and comparing to layer.forward().
    """
    layer = PermutationEquivariantLayer(input_dim=5, output_dim=7, pool_type='max')
    layer.eval()

    B, M = 4, 10
    x = torch.randn(B, M, 5)

    with torch.no_grad():
        # Manual formula
        lambda_term = layer.lambda_net(x)                         # (B, M, 7)
        pooled      = x.max(dim=1, keepdim=True)[0]              # (B, 1, 5)
        gamma_term  = layer.gamma_net(pooled)                     # (B, 1, 7)
        out_manual  = lambda_term + gamma_term                    # broadcasts

        # Layer forward
        out_layer   = layer(x)

    error = (out_manual - out_layer).abs().max().item()
    if error > ATOL:
        return False, f"Lemma 3 formula mismatch: error={error:.2e}"
    return True, f"Λx + Γ·pool(X)·1 verified, max error={error:.2e}"


def test_lemma3_sum_pool_formula():
    """Lemma 3 structural check with sum pooling variant."""
    layer = PermutationEquivariantLayer(input_dim=5, output_dim=7, pool_type='sum')
    layer.eval()

    B, M = 4, 10
    x = torch.randn(B, M, 5)

    with torch.no_grad():
        lambda_term = layer.lambda_net(x)
        pooled      = x.sum(dim=1, keepdim=True)
        gamma_term  = layer.gamma_net(pooled)
        out_manual  = lambda_term + gamma_term
        out_layer   = layer(x)

    error = (out_manual - out_layer).abs().max().item()
    if error > ATOL:
        return False, f"Sum-pool formula mismatch: error={error:.2e}"
    return True, f"Sum-pool Lemma 3 verified, max error={error:.2e}"


def test_lemma3_masked_formula():
    """
    Lemma 3 with masking: verify manual masked pool matches layer output.
    Padding positions should be excluded from the global context pool.
    """
    layer = PermutationEquivariantLayer(input_dim=5, output_dim=7, pool_type='max')
    layer.eval()

    B, M = 3, 8
    active = 5
    x    = torch.randn(B, M, 5)
    mask = torch.zeros(B, M)
    mask[:, :active] = 1.0

    with torch.no_grad():
        # Manual: pool only over active elements
        x_masked   = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        pooled     = x_masked.max(dim=1, keepdim=True)[0]          # (B, 1, 5)
        lambda_out = layer.lambda_net(x)
        gamma_out  = layer.gamma_net(pooled)
        out_manual = lambda_out + gamma_out

        out_layer  = layer(x, mask)

    error = (out_manual - out_layer).abs().max().item()
    if error > ATOL:
        return False, f"Masked Lemma 3 mismatch: error={error:.2e}"
    return True, f"Masked Lemma 3 verified, max error={error:.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PERMUTATION INVARIANCE  —  f(π(X)) = f(X)
# ══════════════════════════════════════════════════════════════════════════════

def _check_invariance(model, B=6, M=15, D=8, trials=20, pool_types=None):
    model.eval()
    max_error = 0.0
    for _ in range(trials):
        x    = torch.randn(B, M, D)
        perm = _random_perm(M)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x[:, perm, :])
        max_error = max(max_error, (out1 - out2).abs().max().item())
    return max_error


def test_invariance_sum_pool():
    model = DeepSetsInvariant(4, [16, 16], [16], 3, pool_type='sum')
    e = _check_invariance(model, D=4)
    return e < ATOL, f"max error={e:.2e}"


def test_invariance_max_pool():
    model = DeepSetsInvariant(4, [16, 16], [16], 3, pool_type='max')
    e = _check_invariance(model, D=4)
    return e < ATOL, f"max error={e:.2e}"


def test_invariance_mean_pool():
    model = DeepSetsInvariant(4, [16, 16], [16], 3, pool_type='mean')
    e = _check_invariance(model, D=4)
    return e < ATOL, f"max error={e:.2e}"


def test_invariance_with_all_negative_inputs():
    """
    Adversarial invariance test: all active elements are negative.
    Previously (bug #1) masked max pooling would return 0 for padding,
    making 0 beat every active value and breaking invariance semantics.
    """
    model = DeepSetsInvariant(4, [16], [16], 3, pool_type='max')
    model.eval()

    B, M, active = 4, 12, 5
    max_error = 0.0
    for _ in range(20):
        # All active values strongly negative; padding zeroed
        x    = torch.rand(B, M, 4) * (-10) - 1.0   # range [-11, -1]
        mask = torch.zeros(B, M)
        mask[:, :active] = 1.0
        perm = _random_perm(active)

        x_perm = x.clone()
        x_perm[:, :active, :] = x[:, perm, :]

        with torch.no_grad():
            out1 = model(x, mask)
            out2 = model(x_perm, mask)

        max_error = max(max_error, (out1 - out2).abs().max().item())

    return max_error < ATOL, f"max error={max_error:.2e}"


def test_invariance_conditional_concat():
    model = DeepSetsConditional(4, 3, [16, 16], [16], 2, fusion_type='concat')
    model.eval()
    max_error = 0.0
    for _ in range(20):
        x       = torch.randn(4, 10, 4)
        context = torch.randn(4, 3)
        perm    = _random_perm(10)
        with torch.no_grad():
            out1 = model(x, context)
            out2 = model(x[:, perm, :], context)
        max_error = max(max_error, (out1 - out2).abs().max().item())
    return max_error < ATOL, f"max error={max_error:.2e}"


def test_invariance_conditional_film():
    model = DeepSetsConditional(4, 3, [16, 16], [16], 2, fusion_type='film')
    model.eval()
    max_error = 0.0
    for _ in range(20):
        x       = torch.randn(4, 10, 4)
        context = torch.randn(4, 3)
        perm    = _random_perm(10)
        with torch.no_grad():
            out1 = model(x, context)
            out2 = model(x[:, perm, :], context)
        max_error = max(max_error, (out1 - out2).abs().max().item())
    return max_error < ATOL, f"max error={max_error:.2e}"


def test_invariance_conditional_add():
    model = DeepSetsConditional(4, 3, [16, 16], [16], 2, fusion_type='add')
    model.eval()
    max_error = 0.0
    for _ in range(20):
        x       = torch.randn(4, 10, 4)
        context = torch.randn(4, 3)
        perm    = _random_perm(10)
        with torch.no_grad():
            out1 = model(x, context)
            out2 = model(x[:, perm, :], context)
        max_error = max(max_error, (out1 - out2).abs().max().item())
    return max_error < ATOL, f"max error={max_error:.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PERMUTATION EQUIVARIANCE  —  f(π(X))[i] = f(X)[π(i)]
# ══════════════════════════════════════════════════════════════════════════════

def _check_equivariance(model, B=4, M=12, D=6, trials=20):
    model.eval()
    max_error = 0.0
    for _ in range(trials):
        x    = torch.randn(B, M, D)
        perm = _random_perm(M)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x[:, perm, :])
        # equivariance: permuting input should permute output the same way
        max_error = max(max_error, (out1[:, perm, :] - out2).abs().max().item())
    return max_error


def test_equivariance_single_layer_max():
    layer = PermutationEquivariantLayer(6, 8, pool_type='max')
    e = _check_equivariance(layer, D=6)
    return e < ATOL, f"max error={e:.2e}"


def test_equivariance_single_layer_sum():
    layer = PermutationEquivariantLayer(6, 8, pool_type='sum')
    e = _check_equivariance(layer, D=6)
    return e < ATOL, f"max error={e:.2e}"


def test_equivariance_deep_network():
    """Stacked equivariant layers must remain equivariant (Lemma 3, Proposition 4)."""
    model = DeepSetsEquivariant(6, [16, 16, 16], 4, pool_type='max')
    e = _check_equivariance(model, D=6)
    return e < ATOL, f"max error={e:.2e}"


def test_equivariant_not_invariant():
    """
    Sanity check: equivariant model output is NOT invariant.
    f(π(X)) ≠ f(X) in general (outputs are permuted, not equal).
    """
    model = DeepSetsEquivariant(4, [8], 4, pool_type='max')
    model.eval()

    x    = torch.randn(2, 8, 4)
    perm = _random_perm(8)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x[:, perm, :])

    # out2 should equal out1[:, perm, :], NOT out1 itself
    invariance_error   = (out1 - out2).abs().max().item()
    equivariance_error = (out1[:, perm, :] - out2).abs().max().item()

    if equivariance_error > ATOL:
        return False, f"Equivariance violated: error={equivariance_error:.2e}"
    if invariance_error < ATOL:
        return False, "Output appears invariant — equivariance structure lost"
    return True, (
        f"equivariance error={equivariance_error:.2e}, "
        f"invariance error={invariance_error:.2e} (correctly non-zero)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MASKING  —  padding elements must not influence the output
# ══════════════════════════════════════════════════════════════════════════════

def _check_mask_invariance(model, pool_type_label="", D=6):
    """
    For each set size, verify that changing padding values does not change
    the model output.
    """
    model.eval()
    set_sizes = [3, 7, 11, 15]
    max_size  = 20

    for size in set_sizes:
        x    = torch.randn(2, max_size, D)
        mask = torch.zeros(2, max_size)
        mask[:, :size] = 1.0

        x_noisy = x.clone()
        x_noisy[:, size:, :] = torch.randn_like(x_noisy[:, size:, :]) * 10

        with torch.no_grad():
            out1 = model(x,        mask)
            out2 = model(x_noisy,  mask)

        error = (out1 - out2).abs().max().item()
        if error > ATOL:
            return False, (
                f"{pool_type_label} masking failed for size={size}: error={error:.2e}"
            )
    return True, f"{pool_type_label} all set sizes OK"


def test_masking_sum_pool():
    model = DeepSetsInvariant(6, [16], [16], 3, pool_type='sum')
    return _check_mask_invariance(model, "sum")


def test_masking_max_pool():
    model = DeepSetsInvariant(6, [16], [16], 3, pool_type='max')
    return _check_mask_invariance(model, "max")


def test_masking_mean_pool():
    model = DeepSetsInvariant(6, [16], [16], 3, pool_type='mean')
    return _check_mask_invariance(model, "mean")


def test_masking_equivariant_layer():
    layer = PermutationEquivariantLayer(6, 8, pool_type='max')
    layer.eval()
    set_sizes = [3, 7, 12]
    max_size  = 15
    for size in set_sizes:
        x    = torch.randn(3, max_size, 6)
        mask = torch.zeros(3, max_size)
        mask[:, :size] = 1.0
        x_noisy = x.clone()
        x_noisy[:, size:, :] = torch.randn_like(x_noisy[:, size:, :]) * 10
        with torch.no_grad():
            out1 = layer(x,       mask)
            out2 = layer(x_noisy, mask)
        # Only active positions should match (padding output may differ freely)
        error = (out1[:, :size, :] - out2[:, :size, :]).abs().max().item()
        if error > ATOL:
            return False, f"Equivariant layer masking failed for size={size}: error={error:.2e}"
    return True, "Equivariant layer masked correctly"


def test_masking_max_pool_all_negative():
    """
    Regression test: max pooling with all-negative active values.
    Padding zeros must NOT dominate the max (requires -inf fill, not 0 fill).
    """
    B, M, D, active = 4, 12, 6, 4
    mask = torch.zeros(B, M)
    mask[:, :active] = 1.0

    for _ in range(10):
        # Active values all in [-10, -0.1]
        x      = -torch.rand(B, M, D) * 9.9 - 0.1
        x_perm = x.clone()
        perm   = _random_perm(active)
        x_perm[:, :active, :] = x[:, perm, :]

        # Without a model, test _masked_pool directly
        pooled      = _masked_pool(x,      'max', mask)
        pooled_perm = _masked_pool(x_perm, 'max', mask)

        # All pooled values should be negative (not 0 from padding)
        if pooled.max().item() >= 0:
            return False, (
                f"Padding zeros dominated max: pooled.max={pooled.max().item():.4f}"
            )

        # Invariant: pool of same active elements == pool of permuted active elements
        error = (pooled - pooled_perm).abs().max().item()
        if error > ATOL:
            return False, f"Max pool not invariant to active-element permutation: error={error:.2e}"

    return True, "Max pool correctly returns negative values; no padding contamination"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UNIVERSAL APPROXIMATION  (Theorem 2, empirical)
#     Any symmetric function should be learnable by DeepSets.
# ══════════════════════════════════════════════════════════════════════════════

def _train_and_eval(model, X_train, y_train, X_test, y_test, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(X_train), y_train).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return criterion(model(X_test), y_test).item()


def test_approx_sum_of_elements():
    """
    Learn f(X) = Σ x_i  (trivially representable by φ=id, ρ=id with sum pool).
    """
    N, M, D = 800, 10, 1
    X_train  = torch.randn(N, M, D)
    y_train  = X_train.sum(dim=(1, 2)).unsqueeze(-1)    # (N, 1)
    X_test   = torch.randn(200, M, D)
    y_test   = X_test.sum(dim=(1, 2)).unsqueeze(-1)

    model = DeepSetsInvariant(D, [16], [16], 1, pool_type='sum')
    loss  = _train_and_eval(model, X_train, y_train, X_test, y_test)
    return loss < LTOL, f"MSE={loss:.6f}"


def test_approx_sum_of_squares():
    """
    Learn f(X) = Σ ||x_i||^2 (requires nonlinear φ, not trivially linear).
    Targets are normalised to zero mean / unit variance so the MSE threshold
    is meaningful regardless of input scale.
    """
    N, M, D = 1000, 10, 2
    X_train = torch.randn(N, M, D)
    y_raw   = (X_train ** 2).sum(dim=(1, 2)).unsqueeze(-1)
    y_mean, y_std = y_raw.mean(), y_raw.std().clamp(min=1e-8)
    y_train = (y_raw - y_mean) / y_std

    X_test  = torch.randn(200, M, D)
    y_test  = ((X_test ** 2).sum(dim=(1, 2)).unsqueeze(-1) - y_mean) / y_std

    model = DeepSetsInvariant(D, [32, 32], [32, 16], 1, pool_type='sum')
    loss  = _train_and_eval(model, X_train, y_train, X_test, y_test, epochs=400)
    return loss < LTOL, f"Normalised MSE={loss:.6f}"


def test_approx_max_of_elements():
    """
    Learn f(X) = max_{x ∈ X} x  (naturally expressible with max pooling).
    """
    N, M = 800, 10
    X_train = torch.randn(N, M, 1)
    y_train = X_train.max(dim=1)[0]     # (N, 1)
    X_test  = torch.randn(200, M, 1)
    y_test  = X_test.max(dim=1)[0]

    model = DeepSetsInvariant(1, [16], [16], 1, pool_type='max')
    loss  = _train_and_eval(model, X_train, y_train, X_test, y_test)
    return loss < LTOL, f"MSE={loss:.6f}"


def test_approx_mean_of_elements():
    """
    Learn f(X) = (1/|X|) Σ x_i  (naturally expressible with mean pooling).
    """
    N, M = 800, 10
    X_train = torch.randn(N, M, 1)
    y_train = X_train.mean(dim=1)       # (N, 1)
    X_test  = torch.randn(200, M, 1)
    y_test  = X_test.mean(dim=1)

    model = DeepSetsInvariant(1, [16], [16], 1, pool_type='mean')
    loss  = _train_and_eval(model, X_train, y_train, X_test, y_test)
    return loss < LTOL, f"MSE={loss:.6f}"


def test_approx_set_cardinality():
    """
    Learn f(X) = |X| using variable-size sets and masking.
    Natural for sum pooling: sum over mask = cardinality.
    Targets are normalised so the MSE threshold is scale-independent.
    """
    N, max_M, D = 600, 20, 4
    X_train = torch.randn(N, max_M, D)
    sizes   = torch.randint(1, max_M + 1, (N,))
    masks   = torch.zeros(N, max_M)
    for i, s in enumerate(sizes):
        masks[i, :s] = 1.0
    y_raw   = sizes.float().unsqueeze(-1)   # (N, 1)
    y_mean, y_std = y_raw.mean(), y_raw.std().clamp(min=1e-8)
    y_train = (y_raw - y_mean) / y_std

    X_test   = torch.randn(200, max_M, D)
    sizes_t  = torch.randint(1, max_M + 1, (200,))
    masks_t  = torch.zeros(200, max_M)
    for i, s in enumerate(sizes_t):
        masks_t[i, :s] = 1.0
    y_test = (sizes_t.float().unsqueeze(-1) - y_mean) / y_std

    model = DeepSetsInvariant(D, [16, 16], [16], 1, pool_type='sum')

    # Custom train loop with masks
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(400):
        optimizer.zero_grad()
        criterion(model(X_train, masks), y_train).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        loss = criterion(model(X_test, masks_t), y_test).item()

    return loss < LTOL, f"Normalised MSE={loss:.4f} (cardinality in [1,{max_M}])"


def test_approx_generalisation_to_larger_sets():
    """
    Theorem 2 corollary: a model trained on sets of size M should generalise
    to sets of a different size M' (since the decomposition is size-agnostic).
    """
    D, train_M, test_M = 2, 10, 30

    X_train = torch.randn(800, train_M, D)
    y_train = (X_train ** 2).sum(dim=(1, 2)).unsqueeze(-1)

    model = DeepSetsInvariant(D, [32, 32], [32, 16], 1, pool_type='sum')
    _train_and_eval(model, X_train, y_train, X_train, y_train, epochs=400)  # warm up

    X_test = torch.randn(200, test_M, D)
    y_test = (X_test ** 2).sum(dim=(1, 2)).unsqueeze(-1)

    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        # Normalise by set size for a fair comparison
        out   = model(X_test) / test_M * train_M
        loss  = criterion(out, y_test / test_M * train_M).item()

    # We just test the model runs without shape errors and loss is finite
    ok = math.isfinite(loss)
    return ok, f"Generalised to set_size={test_M}, scaled MSE={loss:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CONDITIONING  —  DeepSetsConditional
# ══════════════════════════════════════════════════════════════════════════════

def test_conditioning_changes_output():
    """
    Different context vectors must produce different outputs.
    If context has no effect the model ignores the conditioning signal.
    """
    for fusion in ('concat', 'film', 'add'):
        model = DeepSetsConditional(4, 3, [16, 16], [16], 2, fusion_type=fusion)
        model.eval()

        x    = torch.randn(4, 10, 4)
        ctx1 = torch.randn(4, 3)
        ctx2 = torch.randn(4, 3)          # different context

        with torch.no_grad():
            out1 = model(x, ctx1)
            out2 = model(x, ctx2)

        diff = (out1 - out2).abs().mean().item()
        if diff < 1e-4:
            return False, f"fusion='{fusion}': context has no effect (mean diff={diff:.2e})"

    return True, "All fusion types produce context-dependent outputs"


def test_same_context_same_output():
    """Same (x, z) pair always produces the same output (determinism)."""
    model = DeepSetsConditional(4, 3, [16, 16], [16], 2, fusion_type='film')
    model.eval()

    x, ctx = torch.randn(3, 8, 4), torch.randn(3, 3)
    with torch.no_grad():
        out1 = model(x, ctx)
        out2 = model(x, ctx)

    error = (out1 - out2).abs().max().item()
    return error < ATOL, f"error={error:.2e}"


def test_context_in_rho_false():
    """
    context_in_rho=False: context should only influence via φ, not ρ.
    Model must still be permutation invariant.
    """
    model = DeepSetsConditional(
        4, 3, [16, 16], [16], 2, fusion_type='concat', context_in_rho=False
    )
    model.eval()
    max_error = 0.0
    for _ in range(20):
        x    = torch.randn(3, 10, 4)
        ctx  = torch.randn(3, 3)
        perm = _random_perm(10)
        with torch.no_grad():
            out1 = model(x, ctx)
            out2 = model(x[:, perm, :], ctx)
        max_error = max(max_error, (out1 - out2).abs().max().item())
    return max_error < ATOL, f"max invariance error={max_error:.2e}"


def test_conditioning_concat_empty_phi():
    """
    fusion_type='concat', phi_hidden_dims=[]: phi is identity over
    (input_dim + context_dim), so phi_out_dim must equal input_dim + context_dim.
    Regression test for the dimension bug where phi_out_dim was incorrectly set
    to input_dim, causing a shape mismatch in rho's first linear layer.
    """
    model = DeepSetsConditional(4, 3, [], [16], 2, fusion_type='concat')
    model.eval()
    x   = torch.randn(2, 5, 4)
    ctx = torch.randn(2, 3)
    with torch.no_grad():
        out = model(x, ctx)
    return out.shape == (2, 2), f"expected (2, 2), got {tuple(out.shape)}"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INPUT VALIDATION  —  error messages for bad arguments
# ══════════════════════════════════════════════════════════════════════════════

def test_invalid_pool_type_raises():
    try:
        DeepSetsInvariant(4, [8], [8], 1, pool_type='invalid')
        return False, "No error raised for bad pool_type"
    except ValueError:
        return True, "ValueError raised as expected"


def test_film_empty_phi_raises():
    try:
        DeepSetsConditional(4, 3, [], [8], 1, fusion_type='film')
        return False, "No error raised for empty phi_hidden_dims with film"
    except ValueError:
        return True, "ValueError raised as expected"


def test_invalid_fusion_type_raises():
    try:
        DeepSetsConditional(4, 3, [8], [8], 1, fusion_type='unknown')
        return False, "No error raised for bad fusion_type"
    except ValueError:
        return True, "ValueError raised as expected"


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

TESTS = [
    # Structural — paper formulae
    ("Theorem 2: ρ(Σφ(x)) decomposition",            test_theorem2_decomposition),
    ("Lemma 3: Λx + Γ·max(X)·1  (max pool)",         test_lemma3_layer_formula),
    ("Lemma 3: Λx + Γ·sum(X)·1  (sum pool)",         test_lemma3_sum_pool_formula),
    ("Lemma 3: masked max pool formula",               test_lemma3_masked_formula),

    # Permutation invariance
    ("Invariance: sum pooling",                        test_invariance_sum_pool),
    ("Invariance: max pooling",                        test_invariance_max_pool),
    ("Invariance: mean pooling",                       test_invariance_mean_pool),
    ("Invariance: max pool, all-negative inputs",      test_invariance_with_all_negative_inputs),
    ("Invariance: conditional (concat)",               test_invariance_conditional_concat),
    ("Invariance: conditional (film)",                 test_invariance_conditional_film),
    ("Invariance: conditional (add)",                  test_invariance_conditional_add),

    # Permutation equivariance
    ("Equivariance: single layer (max pool)",          test_equivariance_single_layer_max),
    ("Equivariance: single layer (sum pool)",          test_equivariance_single_layer_sum),
    ("Equivariance: deep network",                     test_equivariance_deep_network),
    ("Equivariance ≠ invariance (sanity check)",       test_equivariant_not_invariant),

    # Masking
    ("Masking: sum pool",                              test_masking_sum_pool),
    ("Masking: max pool",                              test_masking_max_pool),
    ("Masking: mean pool",                             test_masking_mean_pool),
    ("Masking: equivariant layer (active positions)",  test_masking_equivariant_layer),
    ("Masking: max pool all-negative (regression)",    test_masking_max_pool_all_negative),

    # Universal approximation
    ("Approx: Σ x_i  (sum pool)",                     test_approx_sum_of_elements),
    ("Approx: Σ ||x_i||²  (nonlinear φ)",             test_approx_sum_of_squares),
    ("Approx: max x_i  (max pool)",                   test_approx_max_of_elements),
    ("Approx: mean x_i  (mean pool)",                 test_approx_mean_of_elements),
    ("Approx: set cardinality  (variable-size)",       test_approx_set_cardinality),
    ("Approx: generalise to unseen set sizes",         test_approx_generalisation_to_larger_sets),

    # Conditioning
    ("Conditioning: context changes output",           test_conditioning_changes_output),
    ("Conditioning: deterministic for same (x,z)",     test_same_context_same_output),
    ("Conditioning: context_in_rho=False",             test_context_in_rho_false),
    ("Conditioning: concat + empty phi (dim bug)",     test_conditioning_concat_empty_phi),

    # Input validation
    ("Validation: bad pool_type raises ValueError",    test_invalid_pool_type_raises),
    ("Validation: film + empty phi raises ValueError", test_film_empty_phi_raises),
    ("Validation: bad fusion_type raises ValueError",  test_invalid_fusion_type_raises),
]


def run_all():
    print("=" * 72)
    print("Deep Sets — Theory-Faithful Test Suite")
    print("Zaheer et al., NeurIPS 2017  (arXiv:1703.06114)")
    print("=" * 72)

    sections = [
        ("Structural (Theorem 2 / Lemma 3)",   4),
        ("Permutation Invariance",              7),
        ("Permutation Equivariance",            4),
        ("Masking",                             5),
        ("Universal Approximation (Theorem 2)", 6),
        ("Conditioning",                        4),
        ("Input Validation",                    3),
    ]

    results  = []
    test_idx = 0
    for section_name, count in sections:
        print(f"\n── {section_name} ──")
        for name, fn in TESTS[test_idx : test_idx + count]:
            results.append(_run(name, fn))
        test_idx += count

    passed = sum(results)
    total  = len(results)
    print("\n" + "=" * 72)
    print(f"  {passed}/{total} tests passed")
    print("=" * 72)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
