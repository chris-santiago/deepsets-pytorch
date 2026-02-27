# Conditional Models

`DeepSetsConditional` processes a set conditioned on an external **context vector** $\mathbf{z}$. This is useful when the task changes based on a query, label, or auxiliary information — for example, answering different questions about the same set depending on context.

The conditional model implements:

$$f(\mathcal{X} \mid \mathbf{z}) = \rho\!\left(\sum_{x \in \mathcal{X}} \varphi(x \mid \mathbf{z})\right)$$

---

## Fusion Strategies

There are three ways to inject context into φ. Choose one via the `fusion_type` parameter.

=== "concat"

    Context is concatenated to each element before the φ network. Simple and effective.

    ```python
    from deepsets import DeepSetsConditional
    import torch

    model = DeepSetsConditional(
        input_dim=10,
        context_dim=5,
        phi_hidden_dims=[32, 32],
        rho_hidden_dims=[32],
        output_dim=3,
        pool_type='sum',
        fusion_type='concat',
    )

    x       = torch.randn(8, 20, 10)   # (B, M, D_x)
    context = torch.randn(8, 5)        # (B, D_z)
    out = model(x, context)
    print(out.shape)   # torch.Size([8, 3])
    ```

    Internally, each element becomes `[x_i || z]` (concatenated along the feature dimension) before being passed to φ. The φ network's effective input dimension is `input_dim + context_dim`.

=== "film"

    **Feature-wise Linear Modulation** (FiLM) applies context as a learned scale-and-shift after the first φ layer:

    $$\mathbf{h} = \varphi_\text{first}(\mathbf{x}_i)$$
    $$\mathbf{h} \leftarrow \text{ReLU}\!\left(\gamma(\mathbf{z}) \odot \mathbf{h} + \beta(\mathbf{z})\right)$$
    $$\varphi(\mathbf{x}_i \mid \mathbf{z}) = \varphi_\text{rest}(\mathbf{h})$$

    FiLM allows the context to **modulate how features are processed**, not just what is input. Requires at least one hidden layer in φ.

    ```python
    model = DeepSetsConditional(
        input_dim=10,
        context_dim=5,
        phi_hidden_dims=[32, 32],   # must be non-empty for film
        rho_hidden_dims=[32],
        output_dim=3,
        fusion_type='film',
    )
    out = model(x, context)   # torch.Size([8, 3])
    ```

=== "add"

    Context is projected to `input_dim` and **added** to each element before the φ network:

    $$\varphi(\mathbf{x}_i \mid \mathbf{z}) = \varphi\!\left(\mathbf{x}_i + W_\text{proj}\,\mathbf{z}\right)$$

    Lightweight and useful when context and elements live in the same conceptual space.

    ```python
    model = DeepSetsConditional(
        input_dim=10,
        context_dim=5,
        phi_hidden_dims=[32],
        rho_hidden_dims=[32],
        output_dim=3,
        fusion_type='add',
    )
    out = model(x, context)   # torch.Size([8, 3])
    ```

---

## The `context_in_rho` Parameter

By default (`context_in_rho=True`), the context vector is also concatenated to the pooled representation before being passed to ρ:

```
rho_input = [pool(φ(X|z)) || z]
```

This gives ρ direct access to the conditioning variable, which helps when the output depends strongly on context independently of the set content.

Set `context_in_rho=False` to exclude the context from ρ:

```python
model = DeepSetsConditional(
    input_dim=10,
    context_dim=5,
    phi_hidden_dims=[32, 32],
    rho_hidden_dims=[32],
    output_dim=3,
    fusion_type='concat',
    context_in_rho=False,   # context only enters through φ
)
```

---

## Choosing a Fusion Strategy

| Strategy | Input to φ | Context capacity | Typical use |
|----------|-----------|-----------------|-------------|
| `concat` | `[x_i \|\| z]` | Moderate | Default choice; always works |
| `film` | `x_i`, then modulated | High | When context should change *how* features are processed |
| `add` | `x_i + Wz` | Low | When context is a simple offset in the same feature space |

!!! tip
    Start with `'concat'` — it's the most expressive for a given hidden-layer size and has no constraints on `phi_hidden_dims`. Switch to `'film'` if you need the context to interact multiplicatively with features.

!!! warning "`film` requires non-empty `phi_hidden_dims`"
    FiLM applies modulation after the first φ layer. If `phi_hidden_dims=[]`, there is no first layer and the model will raise `ValueError`.

---

## Verification: Context Changes Output

```python
ctx_a = torch.randn(8, 5)
ctx_b = torch.randn(8, 5)

with torch.no_grad():
    out_a = model(x, ctx_a)
    out_b = model(x, ctx_b)

print("Outputs differ with different context:", not torch.allclose(out_a, out_b))
# True

# Same context → same output
with torch.no_grad():
    out_c = model(x, ctx_a)
print("Same context → same output:", torch.allclose(out_a, out_c))
# True
```
