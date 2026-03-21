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