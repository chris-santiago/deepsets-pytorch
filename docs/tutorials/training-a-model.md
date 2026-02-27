# Training a Model

In this tutorial you'll build a complete training loop for the **sum-of-digits** task: given a set of integers, predict their sum. This is one of the experiments from the original paper and a clean benchmark for the model's learning capacity.

---

## The Task

- **Input**: a set of `M` integers, each in `[0, 9]` (one-hot encoded as 10-dimensional vectors)
- **Target**: the scalar sum of all integers in the set
- **Why Deep Sets?** The sum is permutation invariant — shuffling the digits doesn't change it.

---

## Step 1 — Data Generation

```python
import torch

def make_batch(batch_size: int, set_size: int):
    """
    Returns:
        x      — (B, M, 10)  one-hot digit vectors
        target — (B, 1)      sum of digits
    """
    digits = torch.randint(0, 10, (batch_size, set_size))  # (B, M)
    x = torch.zeros(batch_size, set_size, 10)
    x.scatter_(2, digits.unsqueeze(-1), 1.0)               # one-hot
    target = digits.float().sum(dim=1, keepdim=True)       # (B, 1)
    return x, target
```

Let's verify:

```python
x, y = make_batch(4, 5)
print(x.shape)   # torch.Size([4, 5, 10])
print(y.shape)   # torch.Size([4, 1])
print(y)         # sums like [[23.], [18.], [31.], [15.]]
```

---

## Step 2 — Model Construction

We use `pool_type='sum'` because the target is itself a sum — the theoretical ideal choice.

```python
from deepsets import DeepSetsInvariant

model = DeepSetsInvariant(
    input_dim=10,
    phi_hidden_dims=[64, 64],
    rho_hidden_dims=[64, 32],
    output_dim=1,
    pool_type='sum',
    dropout=0.0,
)
```

---

## Step 3 — Training Loop

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

SET_SIZE = 10        # sets of 10 digits during training
BATCH_SIZE = 256
N_STEPS = 2000

train_losses = []

model.train()
for step in range(N_STEPS):
    x, target = make_batch(BATCH_SIZE, SET_SIZE)
    pred = model(x)
    loss = loss_fn(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 200 == 0:
        train_losses.append(loss.item())
        print(f"Step {step+1:4d}  MSE loss: {loss.item():.4f}")
```

Expected output (values will vary):

```
Step  200  MSE loss: 4.8312
Step  400  MSE loss: 0.4217
Step  600  MSE loss: 0.0831
Step  800  MSE loss: 0.0213
Step 1000  MSE loss: 0.0072
Step 1200  MSE loss: 0.0031
Step 1400  MSE loss: 0.0018
Step 1600  MSE loss: 0.0011
Step 1800  MSE loss: 0.0008
Step 2000  MSE loss: 0.0006
```

---

## Step 4 — Evaluation

Evaluate on a held-out test set of the same size:

```python
model.eval()
with torch.no_grad():
    x_test, y_test = make_batch(1000, SET_SIZE)
    pred_test = model(x_test)
    test_mse = loss_fn(pred_test, y_test).item()

print(f"Test MSE (set_size={SET_SIZE}): {test_mse:.4f}")
# Test MSE (set_size=10): ~0.001
```

---

## Step 5 — Generalization to Larger Sets

A well-trained Deep Sets model generalizes to larger sets than it was trained on, because the φ–pool–ρ decomposition is size-agnostic:

```python
for test_size in [10, 20, 30, 50]:
    with torch.no_grad():
        x_t, y_t = make_batch(500, test_size)
        mse = loss_fn(model(x_t), y_t).item()
    print(f"  set_size={test_size:2d}  MSE={mse:.4f}")
```

```
  set_size=10  MSE=0.0008
  set_size=20  MSE=0.0019
  set_size=30  MSE=0.0031
  set_size=50  MSE=0.0062
```

MSE grows slightly with set size (larger sums → more absolute error) but the model generalizes correctly — it never saw sets larger than 10 during training.

---

## Step 6 — Plot Training Curve

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
steps = [200 * (i + 1) for i in range(len(train_losses))]
plt.plot(steps, train_losses, marker='o')
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.title("Sum-of-digits — training curve")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
```

---

## What's Next?

- **[Variable-Size Sets](../how-to/variable-size-sets.md)** — Train on batches where set sizes differ per example.
- **[Pooling Strategies](../how-to/pooling-strategies.md)** — Explore when `max` or `mean` pooling outperforms `sum`.
- **[Equivariant Models](../how-to/equivariant-models.md)** — Learn per-element labelling with `DeepSetsEquivariant`.
