# Axiom: Shape-Safe Deep Learning Compiler

Axiom is a domain-specific embedded language (eDSL) built on top of JAX and Flax (`nnx`). It completely eliminates the need for manual `reshape`, `transpose`, and `einsum` boilerplate by introducing **Named-Axis Routing**. 

If your tensor math is structurally unsound, Axiom crashes at compile-time. If it compiles, it runs at the theoretical speed limit of your hardware via JAX's XLA compiler. It is purpose-built for rapid architectural iteration, from custom ODE-based recurrent networks to complex Cross-Attention retrieval pipelines.

---

## 🚀 Key Features

* **Declarative Geometry:** Route, transpose, and project tensors purely by their logical axis names. No more calculating `-1` shapes or tracking index positions.
* **Shape-Safe Arithmetic:** Custom `__add__`, `__mul__`, and `__truediv__` operators guarantee structural alignment before executing XLA math.
* **In-Flight Renaming:** Use the bitwise shift operator (`>>`) to safely bridge mathematical domains (e.g., Sequence Queries to Sequence Keys) directly inside the routing brackets.
* **Packing & Unpacking:** Flatten or split dimensions effortlessly using the `&` operator.
* **Hardware Accelerated:** Fully supports `bfloat16` and Automatic Mixed Precision via explicit `.cast()` boundaries and projection `dtype` arguments.

---

## 📦 Installation

Axiom can be installed directly from GitHub. By default, installing the repository will only pull the core pure-Python packages, allowing it to seamlessly integrate into your existing JAX environment.

To ensure XLA hardware acceleration is configured correctly from scratch, it is highly recommended to explicitly specify your backend during installation.

### Using `uv` (recommended)

Standard installation:

```bash
uv add git+https://github.com/Adam-Jacuch/Axiom.git
```

Explicit CPU backend:

```bash
uv add "axiom[cpu] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

NVIDIA GPU acceleration (CUDA 12):

```bash
uv add "axiom[cuda] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

Bleeding-edge NVIDIA GPU acceleration (CUDA 13):

```bash
uv add "axiom[cuda13] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

AMD GPU / ROCm support:

```bash
uv add "axiom[amd] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

Google Cloud TPU acceleration:

Because Google hosts TPU drivers independently of PyPI, you must install the official JAX TPU wheels directly from Google after installing Axiom.

```bash
# 1. Install Axiom (core)
uv add git+https://github.com/Adam-Jacuch/Axiom.git

# 2. Install Google's TPU drivers
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Using `pip`

Standard installation:

```bash
pip install git+https://github.com/Adam-Jacuch/Axiom.git
```

Explicit CPU backend:

```bash
pip install "axiom[cpu] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

NVIDIA GPU acceleration (CUDA 12):

```bash
pip install "axiom[cuda] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

Bleeding-edge NVIDIA GPU acceleration (CUDA 13):

```bash
pip install "axiom[cuda13] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

AMD GPU / ROCm support:

```bash
pip install "axiom[amd] @ git+https://github.com/Adam-Jacuch/Axiom.git"
```

Google Cloud TPU acceleration:

Because Google hosts TPU drivers independently of PyPI, you must install the official JAX TPU wheels directly from Google after installing Axiom.

```bash
# 1. Install Axiom (core)
pip install git+https://github.com/Adam-Jacuch/Axiom.git

# 2. Install Google's TPU drivers
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```


## 🧠 Mental Model & Syntax

Axiom replaces index-based tracking with **Named-Axis Tokens**. You define your logical dimensions once, bind them to your arrays, and let the compiler handle the heavy lifting of XLA alignments, reshapes, and einsums.

To use Axiom, you only need three main imports:
```python
import jax.numpy as jnp
from axiom import ax, tensor, Module
```

### 1. Axes and Tensors
An `Axis` represents a logical dimension. You access them dynamically from the `ax` namespace. You can optionally define their integer size by calling them.

```python
# Unsized axis (size inferred at runtime)
batch = ax.b 
# Sized axis
dim = ax.d(512) 

# Wrap raw JAX arrays in Axiom metadata
x = tensor(jnp.ones((32, 128, 512)), ax.b, ax.s, ax.d(512))
```

### 2. The Routing Brackets `[...]`
All geometric transformations and network layers are executed inside the tensor indexing brackets. The `->` operator separates the *current* layout from the *desired* layout.

**Safe Transpose**
```python
# Swap sequence and feature dimensions. No dim tracking required!
y = x[ax.b, ax.s, ax.d, "->", ax.b, ax.d, ax.s]

# Ellipsis (...) absorbs unmentioned axes, just like standard NumPy
y = x[..., ax.s, ax.d, "->", ax.d, ax.s, ...]
```

### 3. Packing & Unpacking (`&`)
Use the bitwise AND operator (`&`) to flatten or split physical dimensions effortlessly.

```python
# Pack Heads (h) and Head-Dim (dh) into a single flat dimension
y = x[ax.b, ax.h, ax.dh, "->", ax.b, ax.h & ax.dh]

# Unpack it safely (Axiom automatically infers the missing size if math permits)
z = y[ax.b, ax.h(8) & ax.dh, "->", ax.b, ax.h, ax.dh]
```

### 4. Domain Remapping (`>>`)
Use the right-shift operator (`>>`) to cast logical aliases without moving memory. This is critical for preventing naming collisions (e.g., Query sequence vs. Key sequence in Attention).

```python
# Alias the 'sq' (Sequence Query) axis to 'sk' (Sequence Key)
keys = x[ax.b, (ax.sq >> ax.sk), ax.d]
```

---

## 🏗️ Building Neural Networks

Axiom ops are attached directly to the axes they affect. When the axis token is evaluated in the routing brackets, the operation executes. 

### Implicit Parameters
If you inherit from `axiom.Module` (which wraps Flax `nnx.Module`), Axiom automatically provisions and tracks learnable parameters for affine projections, biases, and normalizations.

```python
class FeedForward(Module):
    def __call__(self, x):
        # 1. Project ax.d to ax.ff
        # 2. Apply SwiGLU activation (automatically halves ax.ff)
        # 3. Project back to ax.d
        h = x[..., ax.d.proj(out=ax.ff(2048)).swiglu()]
        y = h[..., ax.ff.proj(out=ax.d)]
        
        # Apply dropout
        return y[..., ax.d.dropout(rate=0.1)]
```

### Explicit Tensors & Broadcasts
You can easily inject explicit tied weights, biases, or scaling factors. Axiom performs strict named-axis broadcasting automatically.

```python
# RMSNorm with explicit scale tensor
x = x[..., ax.d.norm_rms(scale=scale_tensor)]

# Explicit bias addition
x = x[..., ax.d.bias(tensor=bias_tensor)]
```

### Hardware-Accelerated Attention
Axiom provides a fused `.attend()` op that automatically aligns queries, keys, and values to JAX's strict SDPA backend, flattening batched heads and executing highly optimized flash-attention kernels before restoring your logical layout.

```python
# q has axes [b, h, sq, dh]
# k, v have axes [b, h, sk, dh]
ctx = q[..., ax.sq.attend(keys=k, values=v, dim=ax.dh, is_causal=True)]
```

---

## ⚡ Advanced Capabilities

Axiom v2 includes powerful primitives for expressing cutting-edge models like Mixture of Experts (MoE), State Space Models, and custom decoding loops.

### Symbolic Axis Math
You don't need to pass integer sizes around your architecture. Multiply or divide an unsized axis directly; Axiom evaluates the geometry symbolically at runtime.

```python
# Automatically sizes 'ff' to 4x the physical dimension of 'd' flowing in!
h = x[..., ax.d.proj(out=ax.ff(ax.d * 4)).gelu()]
```

### Pointwise Math & Masking
Perform numerically stable math, masking, and activations directly in the routing stream.

```python
# Causal triangular mask over the Sequence Query / Sequence Key plane
probs = scores[..., ax.sk.mask("tril").softmax()]

# LogSumExp reduction
lse = x[ax.b, ax.s.logsumexp()]
```

### Structural PyTree Operations
While `tensor[...]` strictly enforces 1-to-1 routing, Axiom provides structural methods on the `AxiomTensor` object to combine or split separate memory buffers.

```python
from axiom import AxiomTensor

# Update KV Caches (Concatenate multiple distinct tensors)
updated_cache = AxiomTensor.concat([past_cache, new_states], axis=ax.sq)

# Split fused projections into separate PyTrees
q, k, v = qkv_tensor.split(ax.features, num_chunks=3)

# Top-K Expert Routing for MoE
top_vals, top_idx = gate_scores.topk(ax.e, k=2)
```

---

## 📖 Complete Rosetta Stone Example

Here is how cleanly a full Multi-Head Attention block is expressed in Axiom. Notice the complete lack of `reshape`, `transpose`, or `einsum`.

```python
class Attention(Module):
    def __init__(self, dim: int, heads: int):
        self.dim = dim
        self.heads = heads
        self.dh_size = dim // heads

    def __call__(self, x):
        # Define the packed head block geometry
        head_block = ax.h(self.heads) & ax.dh(self.dh_size)

        # QKV Projections (Notice >> aliases the sequence axis for K and V)
        q = x[..., ax.d.proj(out=head_block, use_bias=False)]
        k = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]
        v = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]

        # Unpack flat [h&dh] into separate [h, dh] dimensions safely
        q = q[ax.b, ax.sq, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sq, ax.dh]
        k = k[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]
        v = v[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]

        # Hardware-Accelerated FlashAttention via Axiom v2
        ctx = q[..., ax.sq.attend(keys=k, values=v, dim=ax.dh, is_causal=True)]

        # Repack heads and apply output projection
        ctx = ctx[..., "->", ax.b, ax.sq, ax.h & ax.dh]
        return ctx[..., (ax.h & ax.dh).proj(out=ax.d(self.dim), use_bias=False)]
```

***

### 15. Axiom v2 Architectures: Structural Ops & Sparse Routing

The core `tensor[...]` syntax is designed for strict, 1-to-1 logical routing and geometric transformations. However, expressing modern, complex architectures—such as Mixture of Experts (MoE), custom autoregressive decoding loops, or novel gated networks—requires operations that alter the physical PyTree structure of the model, scatter discrete states, or evaluate advanced mathematical reductions. 

Axiom v2 introduces a robust suite of operations to handle these advanced architectural needs while strictly maintaining named-axis safety.

#### 15.1 Physical vs. Logical Boundary (Structural Operations)
Logical packing (`&`) and domain remapping (`>>`) are **zero-copy logical view-casts**. They reshape a single tensor's metadata without moving memory or altering the PyTree count.

When an architecture requires combining multiple distinct variables (like appending to a sequence cache) or physically slicing a fused projection into multiple independent variables (like isolating Q, K, and V), Axiom provides **Structural Operations**. Because these break the 1-to-1 routing rule, they are implemented as methods on the `AxiomTensor` itself rather than inside the `tensor[...]` brackets.

* **`AxiomTensor.concat(tensors: List[AxiomTensor], axis: Axis) -> AxiomTensor`**
    Glues multiple distinct PyTrees together. All logical axes outside of the concatenation axis must match exactly.
    ```python
    # Example: Autoregressive state building
    updated_state = AxiomTensor.concat([past_state, new_state], axis=ax.sq)
    ```

* **`AxiomTensor.split(axis: Axis, num_chunks: int) -> Tuple[AxiomTensor, ...]`**
    Physically slices a single tensor into a tuple of distinct tensors.
    ```python
    # Example: Breaking apart a fused dense projection
    q, k, v = qkv_tensor.split(ax.features, num_chunks=3)
    ```

* **`AxiomTensor.topk(axis: Axis, k: int) -> Tuple[AxiomTensor, AxiomTensor]`**
    Returns a tuple of `(values, indices)`. This is the structural backbone for discrete routing decisions.
    ```python
    # Example: Routing to the top 2 experts in an MoE layer
    top_vals, top_idx = gate_scores.topk(ax.e, k=2)
    ```

#### 15.2 Sparse Routing and Gradient Control
To build architectures with discrete routing or custom backward passes, Axiom allows explicit injection and scatter mechanics directly within the routing flow.

* **`Axis.scatter(indices, updates, mode="update|add")`**
    The inverse of `.gather()`. It allows you to safely write specific sequence positions or expert states back into a larger buffer.
* **`Axis.stop_gradient()`**
    Severs the computational graph at the exact moment the token is processed, allowing fine-grained control over what architectural pathways receive gradient updates.

#### 15.3 Advanced Reductions
Reductions inherently consume an axis, returning a `ConsumedSlot`. Beyond standard pooling (`mean`, `sum`), Axiom supports complex mathematical and logical reductions for custom loss functions and masking algorithms.

* **`Axis.logsumexp()`:** Enables the construction of custom, numerically stable attention mechanisms and novel softmax alternatives directly in the routing stream.
* **`Axis.argmax()` / `Axis.argmin()`:** Consumes the target dimension to return discrete indices, bridging the gap between continuous features and discrete logic.
* **`Axis.any()` / `Axis.all()`:** Boolean reductions that pair directly with `.where()` to evaluate complex, multi-dimensional masking conditions.

#### 15.4 Fused Architectural Math (SwiGLU)
Axiom provides standard pointwise operations (`.exp()`, `.log()`, `.abs()`, `.clamp()`, `.rsqrt()`) to enable custom normalizations and activation functions without dropping into raw JAX arrays. 

Furthermore, Axiom treats modern architectural standards as first-class token operations. 

* **`Axis.swiglu(out=None)`**
    Instead of manually routing a split, applying SiLU, and multiplying, `swiglu()` is a native axis operation. It automatically splits the target axis physically in half, applies the SiLU-gate logic, and updates the logical metadata so that the surviving axis is perfectly sized to `N // 2`. You can optionally pass an `out` axis to rename and validate the halved dimension.
    ```python
    # The axis mathematically resolves to half its projected size
    h = x[..., ax.d.proj(out=ax.ff(DIM * 8)).swiglu()]
    
    # Or, explicitly rename the halved axis for clarity
    h = x[..., ax.d.proj(out=ax.ff_inner(DIM * 8)).swiglu(out=ax.ff_out(DIM * 4))]
    ```