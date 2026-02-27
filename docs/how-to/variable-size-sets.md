# Variable-Size Sets

Real datasets rarely have every set the same size. Deep Sets handles this with a **mask** tensor: a float or bool tensor of shape `(B, M)` where `1` marks active elements and `0` marks padding.

---

## Constructing a Mask

Pad all sets in a batch to the length of the longest set, then mark the real elements:

```python
import torch
from deepsets import DeepSetsInvariant

# Suppose set sizes vary: [3, 5, 2, 4]
set_sizes = [3, 5, 2, 4]
max_size = max(set_sizes)         # 5
batch_size = len(set_sizes)
feature_dim = 8

# Allocate padded input tensor (zeros for padding)
x = torch.zeros(batch_size, max_size, feature_dim)
mask = torch.zeros(batch_size, max_size)   # (B, M)

for i, size in enumerate(set_sizes):
    x[i, :size] = torch.randn(size, feature_dim)   # fill real data
    mask[i, :size] = 1.0                            # mark real elements
```

---

## Passing the Mask to the Model

Pass `mask` as the second argument to `forward`:

```python
model = DeepSetsInvariant(
    input_dim=feature_dim,
    phi_hidden_dims=[32, 32],
    rho_hidden_dims=[32],
    output_dim=1,
    pool_type='sum',
)
model.eval()

with torch.no_grad():
    out = model(x, mask)   # (B, 1)
print(out.shape)            # torch.Size([4, 1])
```

All four model types (`DeepSetsInvariant`, `DeepSetsEquivariant`, `PermutationEquivariantLayer`, `DeepSetsConditional`) accept the same `mask` argument.

---

## How Masking Works Internally

The `_masked_pool` utility applies masking differently for each pool type:

=== "sum"

    Padding positions are multiplied by zero before summing:

    ```python
    tensor = tensor * mask.unsqueeze(-1)   # zero out padding
    result = tensor.sum(dim=1)
    ```

=== "max"

    Padding positions are filled with `−∞` so they can never be the maximum:

    ```python
    tensor = tensor.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
    result = tensor.max(dim=1)[0]
    ```

=== "mean"

    Padding is zeroed out and the denominator uses the actual set size:

    ```python
    masked = tensor * mask.unsqueeze(-1)
    count  = mask.sum(dim=1, keepdim=True).clamp(min=1)
    result = masked.sum(dim=1) / count
    ```

---

## Gotcha — Max Pooling with All-Negative Values

If `pool_type='max'` and the active φ-outputs for a set are **all negative**, the max will be a large negative number — which is correct. The `-inf` fill ensures padding positions never "win" even when genuine values are negative.

```python
# All-negative example: verify masking still works correctly
model_max = DeepSetsInvariant(
    input_dim=1, phi_hidden_dims=[4], rho_hidden_dims=[4], output_dim=1,
    pool_type='max',
)
model_max.eval()

# Two sets: sizes 2 and 1
x_neg = torch.tensor([[[-1.0], [-2.0], [0.0]],   # set 1: real=[−1,−2], pad=[0]
                       [[-3.0], [0.0],  [0.0]]])  # set 2: real=[−3],   pad=[0,0]
mask_neg = torch.tensor([[1., 1., 0.], [1., 0., 0.]])

with torch.no_grad():
    out_masked   = model_max(x_neg, mask_neg)
    out_unmasked = model_max(x_neg)         # wrong: padding zeros may win

# out_masked reflects only real elements; out_unmasked may differ
print("masked  :", out_masked.squeeze().tolist())
print("unmasked:", out_unmasked.squeeze().tolist())
```

!!! warning
    Always pass a mask when sets have padding. Without a mask, padded zero-vectors are treated as real elements, which can corrupt the pooled representation — especially for `max` pooling when all active values are negative.

---

## Masking with Equivariant Models

`DeepSetsEquivariant` propagates the mask through every internal `PermutationEquivariantLayer`. The output shape is still `(B, M, output_dim)` — padding positions in the output are undefined and should be ignored downstream.

```python
from deepsets import DeepSetsEquivariant

eq_model = DeepSetsEquivariant(
    input_dim=feature_dim,
    hidden_dims=[32, 32],
    output_dim=4,
    pool_type='max',
)
with torch.no_grad():
    eq_out = eq_model(x, mask)        # (B, M, 4)

# Only use outputs for real positions:
for i, size in enumerate(set_sizes):
    real_out = eq_out[i, :size]       # (size, 4)
```
