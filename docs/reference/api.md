# API Reference

All public symbols are defined in `deepsets.py`. Import them directly:

```python
from deepsets import (
    DeepSetsInvariant,
    PermutationEquivariantLayer,
    DeepSetsEquivariant,
    DeepSetsConditional,
)
```

The internal helper `_masked_pool` is also documented here for completeness.

---

## `_masked_pool`

```python
def _masked_pool(
    tensor: torch.Tensor,
    pool_type: str,
    mask: Optional[torch.Tensor] = None,
    keepdim: bool = False,
) -> torch.Tensor
```

Pool a `(B, M, D)` tensor along the set dimension (`dim=1`) with optional masking.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `Tensor` shape `(B, M, D)` | Input tensor to pool |
| `pool_type` | `str` | One of `'sum'`, `'max'`, `'mean'` |
| `mask` | `Tensor` shape `(B, M)`, optional | Float or bool; `1` = active, `0` = padding |
| `keepdim` | `bool` | If `True`, keep the `M` dimension as size 1 in the output |

**Returns**

`Tensor` of shape `(B, D)` (or `(B, 1, D)` if `keepdim=True`).

**Raises**

`ValueError` if `pool_type` is not `'sum'`, `'max'`, or `'mean'`.

**Notes**

- `sum` / `mean`: padding positions are multiplied by zero.
- `max`: padding positions are filled with `−∞` so they cannot be the maximum even when all active values are negative.
- `mean`: denominator is clamped to `≥ 1` to avoid division by zero on empty sets.

---

## `DeepSetsInvariant`

```python
class DeepSetsInvariant(nn.Module)
```

Deep Sets architecture for **permutation invariant** functions.

Implements Theorem 2: $f(\mathcal{X}) = \rho\!\left(\sum_{x \in \mathcal{X}} \varphi(x)\right)$

φ transforms each element independently; the results are pooled into a single vector and passed through ρ to produce the final output.

### Constructor

```python
DeepSetsInvariant(
    input_dim: int,
    phi_hidden_dims: List[int],
    rho_hidden_dims: List[int],
    output_dim: int,
    pool_type: str = 'sum',
    dropout: float = 0.0,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | — | Dimension of each set element |
| `phi_hidden_dims` | `List[int]` | — | Hidden layer widths for the φ (element-wise) network. Pass `[]` to use a linear φ. |
| `rho_hidden_dims` | `List[int]` | — | Hidden layer widths for the ρ (aggregation) network |
| `output_dim` | `int` | — | Output dimension |
| `pool_type` | `str` | `'sum'` | Aggregation operation: `'sum'`, `'max'`, or `'mean'` |
| `dropout` | `float` | `0.0` | Dropout probability applied after each hidden layer (0 = disabled) |

**Raises**

`ValueError` if `pool_type` is not valid.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `phi` | `nn.Sequential` | Element-wise network |
| `rho` | `nn.Sequential` | Aggregation network |
| `pool_type` | `str` | Pooling mode |

### `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` shape `(B, M, input_dim)` | Batch of sets |
| `mask` | `Tensor` shape `(B, M)`, optional | `1` for active elements, `0` for padding |

**Returns**

`Tensor` of shape `(B, output_dim)`.

### Example

```python
model = DeepSetsInvariant(
    input_dim=10,
    phi_hidden_dims=[64, 64],
    rho_hidden_dims=[64, 32],
    output_dim=1,
    pool_type='sum',
)
x = torch.randn(32, 20, 10)
out = model(x)          # (32, 1)
out = model(x, mask)    # (32, 1), with masking
```

---

## `PermutationEquivariantLayer`

```python
class PermutationEquivariantLayer(nn.Module)
```

Single **permutation equivariant** layer (Lemma 3).

Implements: $f(\mathbf{x}_i) = \Lambda\mathbf{x}_i + \Gamma \cdot \text{pool}(\mathbf{X})$

where $\Lambda$ and $\Gamma$ are learnable linear maps. The pooled global context is broadcast back to every element.

### Constructor

```python
PermutationEquivariantLayer(
    input_dim: int,
    output_dim: int,
    pool_type: str = 'max',
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | — | Input feature dimension per element |
| `output_dim` | `int` | — | Output feature dimension per element |
| `pool_type` | `str` | `'max'` | Pooling for global context: `'max'` or `'sum'` (not `'mean'`) |

**Raises**

`ValueError` if `pool_type` is not `'sum'` or `'max'`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `lambda_net` | `nn.Linear` | Per-element transform Λ: `(input_dim → output_dim)` |
| `gamma_net` | `nn.Linear` | Global-context transform Γ: `(input_dim → output_dim)` |
| `pool_type` | `str` | Pooling mode |

### `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` shape `(B, M, input_dim)` | Batch of sets |
| `mask` | `Tensor` shape `(B, M)`, optional | `1` for active, `0` for padding |

**Returns**

`Tensor` of shape `(B, M, output_dim)`.

### Example

```python
layer = PermutationEquivariantLayer(input_dim=16, output_dim=32, pool_type='max')
x   = torch.randn(4, 10, 16)
out = layer(x)   # (4, 10, 32)
```

---

## `DeepSetsEquivariant`

```python
class DeepSetsEquivariant(nn.Module)
```

Deep Sets architecture for **permutation equivariant** functions.

Stacks `PermutationEquivariantLayer` blocks (with ReLU activations) and optionally applies a final pooling step to produce an invariant output.

### Constructor

```python
DeepSetsEquivariant(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    pool_type: str = 'max',
    final_pool: Optional[str] = None,
    dropout: float = 0.0,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | — | Dimension of each set element |
| `hidden_dims` | `List[int]` | — | Hidden widths for each equivariant layer. The total number of `PermutationEquivariantLayer` instances is `len(hidden_dims) + 1`. |
| `output_dim` | `int` | — | Output dimension per element (or global if `final_pool` is set) |
| `pool_type` | `str` | `'max'` | Pooling inside each equivariant layer: `'sum'` or `'max'` |
| `final_pool` | `str` or `None` | `None` | If set, pool across the set after all layers to produce an invariant `(B, output_dim)` output. One of `'sum'`, `'max'`, `'mean'`. |
| `dropout` | `float` | `0.0` | Dropout probability between hidden layers |

**Raises**

`ValueError` if `final_pool` is set to an invalid value.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | `nn.ModuleList` | All layers: alternating `PermutationEquivariantLayer`, `ReLU`, optional `Dropout` |
| `final_pool` | `str` or `None` | Final pooling mode |

### `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` shape `(B, M, input_dim)` | Batch of sets |
| `mask` | `Tensor` shape `(B, M)`, optional | `1` for active, `0` for padding |

**Returns**

- `Tensor` of shape `(B, M, output_dim)` if `final_pool is None`
- `Tensor` of shape `(B, output_dim)` if `final_pool` is set

### Example

```python
# Set-to-set (equivariant output)
model = DeepSetsEquivariant(
    input_dim=8, hidden_dims=[32, 32], output_dim=4, pool_type='max'
)
out = model(torch.randn(4, 10, 8))   # (4, 10, 4)

# Set-to-vector (invariant output)
model = DeepSetsEquivariant(
    input_dim=8, hidden_dims=[32, 32], output_dim=4,
    pool_type='max', final_pool='sum'
)
out = model(torch.randn(4, 10, 8))   # (4, 4)
```

---

## `DeepSetsConditional`

```python
class DeepSetsConditional(nn.Module)
```

Deep Sets **conditioned** on a context vector $\mathbf{z}$.

Implements: $f(\mathcal{X} \mid \mathbf{z}) = \rho\!\left(\sum_{x \in \mathcal{X}} \varphi(x \mid \mathbf{z})\right)$

Three strategies for fusing context into φ: `'concat'`, `'film'`, `'add'`.

### Constructor

```python
DeepSetsConditional(
    input_dim: int,
    context_dim: int,
    phi_hidden_dims: List[int],
    rho_hidden_dims: List[int],
    output_dim: int,
    pool_type: str = 'sum',
    fusion_type: str = 'concat',
    context_in_rho: bool = True,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | — | Dimension of each set element |
| `context_dim` | `int` | — | Dimension of conditioning vector |
| `phi_hidden_dims` | `List[int]` | — | Hidden widths for φ. Must be non-empty when `fusion_type='film'`. |
| `rho_hidden_dims` | `List[int]` | — | Hidden widths for ρ |
| `output_dim` | `int` | — | Output dimension |
| `pool_type` | `str` | `'sum'` | Pooling: `'sum'`, `'max'`, or `'mean'` |
| `fusion_type` | `str` | `'concat'` | Context fusion strategy: `'concat'`, `'film'`, or `'add'` |
| `context_in_rho` | `bool` | `True` | If `True`, concatenate context to the ρ input so ρ has direct access to $\mathbf{z}$ |

**Raises**

- `ValueError` if `pool_type` is invalid.
- `ValueError` if `fusion_type` is invalid.
- `ValueError` if `fusion_type='film'` and `phi_hidden_dims` is empty.

### Attributes

Depend on `fusion_type`:

**`fusion_type='concat'`**

| Attribute | Description |
|-----------|-------------|
| `phi` | `nn.Sequential` — φ network with input dim `input_dim + context_dim` |
| `rho` | `nn.Sequential` — ρ network |

**`fusion_type='film'`**

| Attribute | Description |
|-----------|-------------|
| `phi_first` | `nn.Linear` — first φ layer: `(input_dim → phi_hidden_dims[0])` |
| `film_gamma` | `nn.Linear` — scale parameter: `(context_dim → phi_hidden_dims[0])` |
| `film_beta` | `nn.Linear` — shift parameter: `(context_dim → phi_hidden_dims[0])` |
| `phi_rest` | `nn.Sequential` — remaining φ layers after FiLM |
| `rho` | `nn.Sequential` — ρ network |

**`fusion_type='add'`**

| Attribute | Description |
|-----------|-------------|
| `context_proj` | `nn.Linear` — project context: `(context_dim → input_dim)` |
| `phi` | `nn.Sequential` — φ network with input dim `input_dim` |
| `rho` | `nn.Sequential` — ρ network |

### `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    context: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` shape `(B, M, input_dim)` | Batch of sets |
| `context` | `Tensor` shape `(B, context_dim)` | Conditioning vectors |
| `mask` | `Tensor` shape `(B, M)`, optional | `1` for active, `0` for padding |

**Returns**

`Tensor` of shape `(B, output_dim)`.

### Static Method: `_make_phi`

```python
@staticmethod
def _make_phi(input_dim: int, hidden_dims: List[int]) -> nn.Sequential
```

Helper that constructs a φ network as `[Linear → ReLU] × n`. Used internally.

### Example

```python
model = DeepSetsConditional(
    input_dim=10,
    context_dim=5,
    phi_hidden_dims=[32, 32],
    rho_hidden_dims=[32],
    output_dim=3,
    fusion_type='film',
    context_in_rho=True,
)

x       = torch.randn(8, 20, 10)
context = torch.randn(8, 5)
out = model(x, context)   # (8, 3)
```
