# Architecture Decisions

This page explains the design choices made in this implementation, and why they were made.

---

## Why `−∞` for Max Pooling with Masks

When computing max pooling over a masked set, we need to ensure that padded (inactive) positions never contribute to the maximum. A naïve approach would be to fill padding with `0`:

```python
# WRONG: if all active values are negative, 0-padded positions will win
tensor[mask == 0] = 0.0
result = tensor.max(dim=1)[0]
```

Consider the case where all active φ-outputs for a set are negative (e.g., `[−0.3, −0.8, −0.1]`) and padding is filled with `0`. The max would incorrectly return `0` (a padding position) instead of `−0.1` (the true maximum among active elements).

The correct approach is to fill padding with $-\infty$:

```python
# CORRECT: -inf never wins against any finite value
tensor = tensor.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
result = tensor.max(dim=1)[0]
```

With $-\infty$ fill, the max over active positions is always correct, even when all values are negative. This is implemented in `_masked_pool`:

```python
elif pool_type == 'max':
    if mask is not None:
        tensor = tensor.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
    return tensor.max(dim=1, keepdim=keepdim)[0]
```

!!! warning "Downstream NaN risk"
    If a set has **no active elements** (all-zero mask), the max over `−∞` values returns `−∞`. Downstream layers receiving `−∞` may produce `NaN` gradients. Always ensure at least one active element per set in your data.

---

## `nn.ModuleList` vs `nn.Sequential` in `DeepSetsEquivariant`

`DeepSetsEquivariant` uses `nn.ModuleList` to store its layers rather than `nn.Sequential`:

```python
self.layers = nn.ModuleList(layers)
```

The `forward` method then iterates manually:

```python
for layer in self.layers:
    if isinstance(layer, PermutationEquivariantLayer):
        out = layer(out, mask)   # pass mask
    else:
        out = layer(out)         # ReLU / Dropout — no mask arg
```

**Why not `nn.Sequential`?**

`nn.Sequential.forward` calls each layer with a single argument. `PermutationEquivariantLayer.forward` requires two arguments: the tensor `x` and the optional `mask`. There is no standard way to thread extra arguments through `nn.Sequential`.

`nn.ModuleList` registers all submodules for parameter tracking (so `model.parameters()` and `model.to(device)` work correctly) while giving full control over the `forward` pass.

---

## The φ–Pool–ρ Decomposition in Practice

The theoretical decomposition $f(\mathcal{X}) = \rho(\sum_x \varphi(x))$ has a natural implementation trade-off: how deep should φ and ρ be?

**Recommended heuristic** (from the paper and empirical practice):

| Network | Depth | Reasoning |
|---------|-------|-----------|
| φ | Deeper (3–4 layers) | Needs to learn rich per-element representations before pooling discards order information |
| ρ | Shallower (2–3 layers) | Processes a fixed-size pooled vector; standard MLP task |

Pooling is an information bottleneck: anything φ doesn't capture is lost. Investing capacity in φ before the bottleneck is generally more efficient than in ρ after it.

---

## FiLM Conditioning — Design Rationale

The `'film'` fusion strategy implements **Feature-wise Linear Modulation** (Perez et al., 2018):

$$\mathbf{h} = \text{ReLU}(\gamma(\mathbf{z}) \odot \varphi_\text{first}(\mathbf{x}_i) + \beta(\mathbf{z}))$$

FiLM applies context as a **multiplicative and additive modulation** of intermediate features, which is strictly more expressive than simple concatenation or addition:

| Fusion | Context influence | Expressiveness | Parameters added |
|--------|------------------|---------------|-----------------|
| `add` | Shifts input space | Low | `context_dim × input_dim` |
| `concat` | Extends input | Moderate | `context_dim × phi_hidden_dims[0]` |
| `film` | Scales + shifts hidden features | High | `2 × context_dim × phi_hidden_dims[0]` |

The reason FiLM requires `phi_hidden_dims` to be non-empty is that it modulates the output of the *first φ layer*. Without any hidden layer, there is no intermediate representation to modulate.

---

## Why `pool_type='mean'` is Excluded from `PermutationEquivariantLayer`

Lemma 3 characterises equivariant layers as $\Lambda x_i + \Gamma \cdot \text{pool}(X)$ where pool is a **fixed** global aggregation. Mean pooling divides by the set size $|\mathcal{X}|$, which is a function of the input — not a fixed symmetric function. This breaks the theoretical characterisation and can cause instability when set sizes vary within a batch.

For this reason, `PermutationEquivariantLayer` only supports `'sum'` and `'max'`, both of which have clean theoretical properties in the equivariant layer formulation.

---

## Comparison to Alternative Approaches

| Method | Time complexity | Permutation invariant | Universal approximation | Notes |
|--------|----------------|----------------------|------------------------|-------|
| **Deep Sets** | $O(M)$ | ✓ (exact) | ✓ (Theorem 2) | Linear in set size |
| Pairwise coupling | $O(M^2)$ | ✓ (exact) | ✓ | Expensive for large sets |
| Sort + RNN | $O(M \log M)$ | ✗ (approximate) | — | Sort not differentiable |
| Attention (naive) | $O(M^2)$ | ✓ (with symmetric attn) | ✓ | Quadratic memory |
| Graph neural networks | $O(M + E)$ | ✓ (with global pool) | ✓ | Requires graph structure |

Deep Sets is the **only method** that achieves exact permutation invariance with $O(M)$ complexity and a universal approximation guarantee, making it ideal for large sets where pairwise methods are prohibitive.

---

## Pooling Mathematics

For a set $\mathcal{X} = \{x_1, \ldots, x_M\}$ with φ-outputs $\mathbf{h}_i = \varphi(x_i) \in \mathbb{R}^d$:

$$\text{sum}: \quad \mathbf{s} = \sum_{i=1}^{M} \mathbf{h}_i$$

$$\text{max}: \quad \mathbf{s}_j = \max_{i=1}^{M} (\mathbf{h}_i)_j \quad \text{(element-wise)}$$

$$\text{mean}: \quad \mathbf{s} = \frac{1}{M}\sum_{i=1}^{M} \mathbf{h}_i$$

With masking (active set $\mathcal{A} \subseteq \{1,\ldots,M\}$):

$$\text{sum}: \quad \mathbf{s} = \sum_{i \in \mathcal{A}} \mathbf{h}_i$$

$$\text{max}: \quad \mathbf{s}_j = \max_{i \in \mathcal{A}} (\mathbf{h}_i)_j \quad \text{(padding filled with} -\infty\text{)}$$

$$\text{mean}: \quad \mathbf{s} = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \mathbf{h}_i$$
