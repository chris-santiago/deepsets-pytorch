# Equivariant Models

A **permutation equivariant** function maps a set to a set of the same size, preserving the ordering relationship: if you permute the input, the output is permuted identically. This is useful for tasks where you need a per-element output — anomaly scores, element labels, or intermediate representations.

---

## The Building Block: `PermutationEquivariantLayer`

A single equivariant layer implements Lemma 3 from the paper:

$$f(\mathbf{x}_i) = \Lambda \mathbf{x}_i + \Gamma \cdot \text{pool}(\mathbf{X})$$

Each element is transformed by its own features (via $\Lambda$) **plus** a global context signal (via $\Gamma$, broadcast from pooling). This shared context is what makes the layer equivariant rather than just element-wise.

```python
import torch
from deepsets import PermutationEquivariantLayer

layer = PermutationEquivariantLayer(
    input_dim=8,
    output_dim=16,
    pool_type='max',   # 'max' or 'sum'
)
layer.eval()

x = torch.randn(4, 10, 8)   # (batch=4, set_size=10, features=8)
out = layer(x)
print(out.shape)              # torch.Size([4, 10, 16])
```

### Verifying Equivariance

```python
perm = torch.randperm(10)
with torch.no_grad():
    out_orig = layer(x)
    out_perm = layer(x[:, perm, :])

# Output permuted by the same perm should equal output of permuted input
max_err = (out_orig[:, perm, :] - out_perm).abs().max().item()
print(f"Equivariance error: {max_err:.2e}")  # ~0.00e+00
```

---

## Stacking Layers: `DeepSetsEquivariant`

`DeepSetsEquivariant` stacks multiple `PermutationEquivariantLayer` blocks with ReLU activations between them. A final `PermutationEquivariantLayer` (no trailing ReLU) produces the output.

```python
from deepsets import DeepSetsEquivariant

model = DeepSetsEquivariant(
    input_dim=8,
    hidden_dims=[32, 32],   # 2 hidden equivariant layers + 1 output layer = 3 total
    output_dim=1,
    pool_type='max',
)
model.eval()

x = torch.randn(4, 10, 8)
out = model(x)
print(out.shape)   # torch.Size([4, 10, 1]) — one score per element
```

### With Dropout

```python
model = DeepSetsEquivariant(
    input_dim=8,
    hidden_dims=[64, 64, 32],
    output_dim=4,
    pool_type='max',
    dropout=0.1,
)
```

---

## Getting an Invariant Output with `final_pool`

If your downstream task needs a single vector per set (invariant output) but you still want equivariant intermediate representations, set `final_pool`:

```python
model_inv = DeepSetsEquivariant(
    input_dim=8,
    hidden_dims=[32, 32],
    output_dim=16,
    pool_type='max',
    final_pool='sum',   # 'sum', 'max', or 'mean'
)
out = model_inv(x)
print(out.shape)   # torch.Size([4, 16]) — one vector per set
```

The `final_pool` collapses the set dimension after all equivariant layers:

```
x         → [PermEqLayer → ReLU] × n → PermEqLayer → _masked_pool → output
(B,M,D_in)                                           (B,M,D_out)   (B,D_out)
```

---

## Anomaly Detection Example

Score each element in a set for how "unusual" it is:

```python
model = DeepSetsEquivariant(
    input_dim=16,
    hidden_dims=[64, 64],
    output_dim=1,       # one anomaly score per element
    pool_type='max',
)

x = torch.randn(32, 20, 16)   # batch of 32 sets, 20 elements each
scores = model(x)              # (32, 20, 1)
scores = scores.squeeze(-1)    # (32, 20)

# Highest-scoring element is the predicted anomaly
anomaly_idx = scores.argmax(dim=1)
print(anomaly_idx.shape)   # torch.Size([32])
```

---

## Using a Mask with Equivariant Models

Pass a mask the same way as with invariant models:

```python
mask = torch.ones(4, 10)
mask[0, 7:] = 0   # first set has only 7 elements

out = model(x, mask)   # (4, 10, output_dim)

# Only positions [:7] are meaningful for the first set
```

!!! note
    When `final_pool` is set, the mask is also passed to `_masked_pool` for the final aggregation, so padding elements don't affect the invariant output.
