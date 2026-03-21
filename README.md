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

**For standard usage (via GitHub):**
```bash
pip install git+[https://github.com/Adam-Jacuch/Axiom.git](https://github.com/Adam-Jacuch/Axiom.git)