# How-to Guides

How-to guides are **task-oriented**: they answer the question "how do I…?" with focused, practical recipes. They assume you already understand the basics.

---

## Available Guides

### [Variable-Size Sets](variable-size-sets.md)
How to construct masks, pass them to any model, and avoid the max-pooling pitfall when all values are negative.

### [Pooling Strategies](pooling-strategies.md)
How to choose between sum, max, and mean pooling for your task, with a practical decision guide and comparison table.

### [Equivariant Models](equivariant-models.md)
How to build set-to-set models using `PermutationEquivariantLayer` and `DeepSetsEquivariant`, including stacking layers and combining with a final invariant pool.

### [Conditional Models](conditional-models.md)
How to use `DeepSetsConditional` with each fusion strategy (`concat`, `film`, `add`) and when to use `context_in_rho`.

---

!!! note "Looking for API details?"
    See the [API Reference](../reference/api.md) for complete parameter tables and return-type documentation.
