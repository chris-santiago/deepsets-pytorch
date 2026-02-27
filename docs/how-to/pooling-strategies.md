# Pooling Strategies

Every Deep Sets model has a `pool_type` parameter that controls how individual element representations are aggregated into a single set-level vector. Choosing the right pooling strategy can noticeably affect performance.

---

## Available Options

| `pool_type` | Formula | Key property |
|-------------|---------|--------------|
| `'sum'` | $\sum_{x \in \mathcal{X}} \varphi(x)$ | Sensitive to set size; can learn counting |
| `'max'` | $\max_{x \in \mathcal{X}} \varphi(x)$ (element-wise) | Sensitive to extreme values; size-invariant |
| `'mean'` | $\frac{1}{\|\mathcal{X}\|}\sum_{x \in \mathcal{X}} \varphi(x)$ | Normalized; robust to set-size variation |

---

## Decision Guide

### Use `sum` when:
- The target is **additive** in the elements (e.g., total count, sum of values, total energy)
- You want the model to be sensitive to **how many** elements satisfy a condition
- You are working with fixed-size sets (no normalization needed)

```python
# Sum-of-digits example: target = Σ digits
model = DeepSetsInvariant(..., pool_type='sum')
```

### Use `max` when:
- The target depends on **extreme** or **salient** elements (e.g., "does the set contain an outlier?")
- You want **size-invariant** representations (the maximum doesn't grow with |X|)
- Building equivariant models (only `'sum'` and `'max'` are supported in `PermutationEquivariantLayer`)

```python
# Anomaly detection: does the set contain a high-value element?
model = DeepSetsInvariant(..., pool_type='max')
```

### Use `mean` when:
- The target is a **statistical average** (e.g., sample mean, proportion)
- Set sizes vary widely and you want normalization built in
- You do not want the model's scale to grow with set size

```python
# Estimate average feature value across a variable-size population
model = DeepSetsInvariant(..., pool_type='mean')
```

---

## Comparison Table

| Task | Recommended pool | Reason |
|------|-----------------|--------|
| Sum of digits | `sum` | Target is additive |
| Max value in set | `max` | Target is extreme |
| Mean of features | `mean` | Target is average |
| Set classification | `max` or `sum` | Depends on discriminative features |
| Anomaly / outlier detection | `max` | Extreme elements drive decision |
| Cardinality estimation | `sum` | Sensitive to count |
| Point cloud classification | `max` | Robust to density variation |
| Population statistics | `mean` | Normalized aggregation |

---

## Pooling in Equivariant Layers

`PermutationEquivariantLayer` only supports `'sum'` and `'max'` (not `'mean'`). This restriction comes from the theoretical form of Lemma 3 — see [Deep Sets Theory](../explanation/theory.md) for details.

```python
from deepsets import PermutationEquivariantLayer

# OK
layer = PermutationEquivariantLayer(input_dim=16, output_dim=16, pool_type='max')

# Raises ValueError
layer = PermutationEquivariantLayer(input_dim=16, output_dim=16, pool_type='mean')
```

---

## Final Pooling in `DeepSetsEquivariant`

When using `DeepSetsEquivariant`, there are **two** pooling parameters:

- **`pool_type`**: pooling used inside each equivariant layer for global context broadcast
- **`final_pool`**: optional pooling applied after all layers to collapse `(B, M, D)` → `(B, D)`

These can be different:

```python
from deepsets import DeepSetsEquivariant

model = DeepSetsEquivariant(
    input_dim=16,
    hidden_dims=[32, 32],
    output_dim=8,
    pool_type='max',       # max inside each equivariant layer
    final_pool='sum',      # sum to get invariant output
)
```

---

## Practical Tips

!!! tip "Start with `sum`"
    For most regression tasks on sets, `sum` pooling converges fastest and achieves the lowest loss. Switch to `max` if you suspect the signal lives in extreme elements.

!!! tip "Variable sizes → prefer `mean`"
    If your sets vary drastically in size (e.g., 2 to 200 elements), `mean` normalizes the pooled representation so the model doesn't need to learn a size correction.

!!! tip "When in doubt, compare all three"
    Training three models with identical architecture but different `pool_type` values is cheap and often reveals which inductive bias best matches your data.
