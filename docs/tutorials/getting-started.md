# Getting Started

This tutorial walks you through installing the library, creating your first model, and verifying that it really is permutation invariant.

---

## Prerequisites

- Python 3.8+
- PyTorch 1.10+

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd deepsets
pip install torch
```

If you want to build the documentation site:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

---

## Step 1 — Import

```python
import torch
from deepsets import DeepSetsInvariant
```

---

## Step 2 — Create a Model

`DeepSetsInvariant` takes a batch of sets `(B, M, D)` and produces one output vector per set `(B, output_dim)`.

```python
model = DeepSetsInvariant(
    input_dim=4,          # each set element has 4 features
    phi_hidden_dims=[32, 32],   # φ network: 2 hidden layers
    rho_hidden_dims=[32],       # ρ network: 1 hidden layer
    output_dim=1,               # predict a scalar per set
    pool_type='sum',            # pooling function
)
model.eval()
print(model)
```

```
DeepSetsInvariant(
  (phi): Sequential(Linear(4, 32), ReLU(), Linear(32, 32), ReLU())
  (rho): Sequential(Linear(32, 32), ReLU(), Linear(32, 1))
)
```

---

## Step 3 — Run a Forward Pass

Create a random batch of 8 sets, each containing 10 elements of dimension 4:

```python
x = torch.randn(8, 10, 4)   # (batch=8, set_size=10, features=4)
with torch.no_grad():
    out = model(x)
print(out.shape)  # torch.Size([8, 1])
```

---

## Step 4 — Verify Permutation Invariance

The defining property of Deep Sets is that the output is unchanged when you shuffle the elements of a set. Let's check this numerically:

```python
import torch

# Build a random permutation of the 10 elements
perm = torch.randperm(10)
x_perm = x[:, perm, :]     # permute the set dimension

with torch.no_grad():
    out_orig = model(x)
    out_perm = model(x_perm)

max_error = (out_orig - out_perm).abs().max().item()
print(f"Max absolute difference: {max_error:.2e}")
# Max absolute difference: 0.00e+00
```

The error is exactly zero (up to floating-point precision) because the pooling operation — summing the φ outputs — is inherently order-independent.

---

## Step 5 — Try Different Pooling

The `pool_type` argument controls how individual element representations are aggregated:

=== "sum"

    ```python
    model_sum = DeepSetsInvariant(
        input_dim=4, phi_hidden_dims=[32], rho_hidden_dims=[32], output_dim=1,
        pool_type='sum'
    )
    ```

=== "max"

    ```python
    model_max = DeepSetsInvariant(
        input_dim=4, phi_hidden_dims=[32], rho_hidden_dims=[32], output_dim=1,
        pool_type='max'
    )
    ```

=== "mean"

    ```python
    model_mean = DeepSetsInvariant(
        input_dim=4, phi_hidden_dims=[32], rho_hidden_dims=[32], output_dim=1,
        pool_type='mean'
    )
    ```

See [Pooling Strategies](../how-to/pooling-strategies.md) for guidance on which to choose.

---

## What's Next?

- **[Training a Model](training-a-model.md)** — Build a full training loop on the sum-of-digits task.
- **[Variable-Size Sets](../how-to/variable-size-sets.md)** — Handle batches where sets have different numbers of elements.
