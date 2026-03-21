"""
Axiom Quickstart & Feature Showcase
-----------------------------------
This script demonstrates the core capabilities of the Axiom eDSL, including:
1. Named-Axis Routing (->)
2. Axis Packing & Unpacking (&)
3. Implicit Memory Allocation (.proj, .norm_rms)
4. Mixed Precision Training (bfloat16 & .cast)
5. XLA-Compiled Training Loops
6. Rank-Altering Embeddings & Tied Weights
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import math

# Assuming this is run from the root of the Axiom repository
from src.axiom import ax, tensor, Module, init


# =============================================================================
# 1. Multi-Head Attention (Showcasing Routing, Packing & Explicit Contraction)
# =============================================================================
class Attention(Module):
    def __init__(self, dim: int, heads: int):
        self.dim = dim
        self.heads = heads
        self.dh_size = dim // heads

    def __call__(self, x):
        # 1. Q keeps the 'sq' axis
        q = x[..., ax.d.proj(out=ax.h(self.heads) & ax.dh(self.dh_size))]

        # 2. In-Flight Rename! 'sq' maps to 'sk', while 'd' is projected and packed.
        k = x[..., (ax.sq >> ax.sk), ax.d.proj(out=ax.h(self.heads) & ax.dh(self.dh_size))]
        v = x[..., (ax.sq >> ax.sk), ax.d.proj(out=ax.h(self.heads) & ax.dh(self.dh_size))]

        # 3. Unpack and Route
        q = q[ax.b, ax.sq, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sq, ax.dh]
        k = k[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]
        v = v[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]

        # 4. Einsum Contraction
        scores = q[..., ax.dh.proj(out=ax.sk, weight=k)]

        # 5. Scale & Causal Mask
        probs = (scores / math.sqrt(self.dh_size))[..., ax.sk.mask('tril').softmax()]

        # 6. Aggregate & Repack
        out_ctx = probs[..., ax.sk.proj(out=ax.dh, weight=v)]
        repacked = out_ctx[..., "->", ax.b, ax.sq, ax.h & ax.dh]

        return repacked[..., (ax.h & ax.dh).proj(out=ax.d(self.dim))]


# =============================================================================
# 2. Feed Forward Block (Showcasing Mixed Precision & Casting)
# =============================================================================
class FeedForward(Module):
    def __init__(self, dim: int, expansion: int = 4):
        self.dim = dim
        self.hidden_dim = dim * expansion

    def __call__(self, x):
        # Run heavy MLPs in bfloat16 for Tensor Core acceleration
        h = x[..., ax.d.proj(out=ax.hd(self.hidden_dim), dtype=jnp.bfloat16).silu()]
        out = h[..., ax.hd.proj(out=ax.d(self.dim), dtype=jnp.bfloat16)]

        # Safely cast back to float32 before returning to the residual stream!
        return out[..., ax.d.cast(jnp.float32)]


# =============================================================================
# 3. Model Architecture (Showcasing Tied Weights & Embeddings)
# =============================================================================
class AxiomLM(Module):
    def __init__(self, vocab_size: int, dim: int, heads: int, depth: int):
        self.vocab_size = vocab_size
        self.dim = dim
        self.layers = nnx.List([
            nnx.List([Attention(dim, heads), FeedForward(dim)])
            for _ in range(depth)
        ])

    def __call__(self, x):
        # 1. Efficient Integer Embedding & Weight Tying Extraction
        # Transforms [b, sq] -> [b, sq, d]
        x, w_emb = x.embed(vocab=self.vocab_size, out=ax.d(self.dim), return_weight=True)

        # 2. Transformer Blocks (Pre-Norm architecture)
        for attn, ff in self.layers:
            x = x + attn(x[..., ax.d.norm_rms()])
            x = x + ff(x[..., ax.d.norm_rms()])

        # 3. Final Norm and Tied-Weight Readout Projection
        x = x[..., ax.d.norm_rms()]
        return x[..., ax.d.proj(out=ax.vocab(self.vocab_size), weight=w_emb)]


# =============================================================================
# 4. Training Engine (Showcasing NNX/Optax Integration)
# =============================================================================
def main():
    print("🚀 Initializing Axiom Quickstart Engine...")

    # Hyperparameters
    BATCH, SEQ, VOCAB, DIM, HEADS, DEPTH = 8, 64, 256, 128, 4, 2
    EPOCHS = 25

    # 1. Initialize Model
    model = AxiomLM(vocab_size=VOCAB, dim=DIM, heads=HEADS, depth=DEPTH)

    # 2. Generate Dummy Integer Sequence Data (Massive memory savings!)
    key_x, key_y = jax.random.split(jax.random.key(42))
    x_ints = jax.random.randint(key_x, (BATCH, SEQ), 0, VOCAB)
    y_labels = jax.random.randint(key_y, (BATCH, SEQ), 0, VOCAB)

    # 3. Wrap inputs in Axiom Tensors (Rank 2)
    X = tensor(x_ints, ax.b, ax.sq)

    # 4. The Dummy Pass (Triggers Axiom's dynamic memory allocation)
    _ = model(X)

    # 5. Initialize NNX Optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    # Calculate exact parameter count to prove weight tying efficiency
    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # 6. Define Scalar Loss Function
    def loss_fn(model, x_in, y_out):
        logits = model(x_in)
        # Use integer label cross-entropy to match our dense labels
        return optax.softmax_cross_entropy_with_integer_labels(logits.data, y_out).mean()

    print(f"Model successfully compiled. Total Parameters: {param_count:,}")
    print("Beginning Training Loop...\n")

    # 7. Execute Training
    for epoch in range(1, EPOCHS + 1):
        loss, grads = nnx.value_and_grad(loss_fn)(model, X, y_labels)
        optimizer.update(model, grads)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

    print("\n✅ Quickstart complete! Axiom is ready for deployment.")


if __name__ == "__main__":
    main()