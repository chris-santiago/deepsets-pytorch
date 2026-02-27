# Explanation

Explanation pages are **understanding-oriented**: they provide background, theory, and design rationale. They answer the question "why?" rather than "how?".

---

## Available Explanations

### [Deep Sets Theory](theory.md)
Formal statements of Theorem 2 (permutation invariance + universal approximation) and Lemma 3 (permutation equivariance), intuitive explanations, and proof sketches.

### [Architecture Decisions](architecture.md)
Design rationale behind the implementation: why `_masked_pool` uses `−∞` for max pooling, why `nn.ModuleList` is used instead of `nn.Sequential` in `DeepSetsEquivariant`, the mathematics of FiLM conditioning, and a comparison of Deep Sets to alternative approaches.
