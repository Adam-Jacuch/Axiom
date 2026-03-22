"""
Axiom Quickstart & Rosetta Stone
--------------------------------
This script demonstrates:

1. Named-axis routing (->)
2. Axis packing and unpacking (&)
3. Safe domain remapping (>>)
4. Implicit parameter allocation (.proj, .norm_rms, .gate)
5. Explicit parameter injection (tied weights)
6. Structured masking with .mask("tril")
7. Mixed precision with bfloat16 and .cast(...)
8. Rank-altering embeddings and tied readout
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from src.axiom import ax, tensor, Module


# =============================================================================
# 1. Multi-Head Attention
# =============================================================================
class Attention(Module):
    def __init__(self, dim: int, heads: int):
        self.dim = dim
        self.heads = heads
        self.dh_size = dim // heads

    def __call__(self, x):
        head_block = ax.h(self.heads) & ax.dh(self.dh_size)

        # Attention projections are commonly bias-free, so we say so explicitly.
        q = x[..., ax.d.proj(out=head_block, use_bias=False)]
        k = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]
        v = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]

        # Unpack [h&dh] -> [h, dh]
        q = q[ax.b, ax.sq, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sq, ax.dh]
        k = k[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]
        v = v[ax.b, ax.sk, ax.h(self.heads) & ax.dh, "->", ax.b, ax.h, ax.sk, ax.dh]

        # Scores: [b, h, sq, sk]
        scores = q[..., ax.dh.proj(out=ax.sk, weight=k, use_bias=False)]
        scores = scores / math.sqrt(self.dh_size)

        # Causal masking on the (sq, sk) plane
        probs = scores[..., ax.sk.mask("tril").softmax()]

        # Aggregate values
        ctx = probs[..., ax.sk.proj(out=ax.dh, weight=v, use_bias=False)]

        # Repack heads
        ctx = ctx[..., "->", ax.b, ax.sq, ax.h & ax.dh]

        # Output projection
        return ctx[..., (ax.h & ax.dh).proj(out=ax.d(self.dim), use_bias=False)]


# =============================================================================
# 2. Feed Forward
# =============================================================================
class FeedForward(Module):
    def __init__(self, dim: int, expansion: int = 4):
        self.dim = dim
        self.hidden_dim = dim * expansion

    def __call__(self, x):
        # Here we intentionally use default bias=True for brevity.
        h = x[..., ax.d.proj(out=ax.hd(self.hidden_dim), dtype=jnp.bfloat16).silu()]
        y = h[..., ax.hd.proj(out=ax.d(self.dim), dtype=jnp.bfloat16)]

        # Cast back before rejoining the residual stream
        return y[..., ax.d.cast(jnp.float32)]


# =============================================================================
# 3. Language Model
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
        # Integer tokens [b, sq] -> embeddings [b, sq, d]
        x, w_emb = x.embed(vocab=self.vocab_size, out=ax.d(self.dim), return_weight=True)

        # Example of an implicit learnable residual gate
        x = x[..., ax.d.gate()]

        # Pre-norm transformer blocks
        for attn, ff in self.layers:
            x = x + attn(x[..., ax.d.norm_rms()])
            x = x + ff(x[..., ax.d.norm_rms()])

        x = x[..., ax.d.norm_rms()]

        # Tied readout
        return x[..., ax.d.proj(out=ax.vocab(self.vocab_size), weight=w_emb, use_bias=False)]


# =============================================================================
# 4. Training Loop
# =============================================================================
def main():
    print("🚀 Initializing Axiom Quickstart Engine...")

    BATCH, SEQ, VOCAB, DIM, HEADS, DEPTH = 8, 64, 256, 128, 4, 2
    EPOCHS = 25

    model = AxiomLM(vocab_size=VOCAB, dim=DIM, heads=HEADS, depth=DEPTH)

    key_x, key_y = jax.random.split(jax.random.key(42))
    x_ints = jax.random.randint(key_x, (BATCH, SEQ), 0, VOCAB)
    y_labels = jax.random.randint(key_y, (BATCH, SEQ), 0, VOCAB)

    X = tensor(x_ints, ax.b, ax.sq)

    # First pass allocates implicit parameters
    _ = model(X)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))

    def loss_fn(model, x_in, y_out):
        logits = model(x_in)
        return optax.softmax_cross_entropy_with_integer_labels(logits.data, y_out).mean()

    print(f"Model successfully initialized. Total parameters: {param_count:,}")
    print("Beginning training loop...\n")

    for epoch in range(1, EPOCHS + 1):
        loss, grads = nnx.value_and_grad(loss_fn)(model, X, y_labels)
        optimizer.update(model, grads)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

    print("\n✅ Quickstart complete.")


if __name__ == "__main__":
    main()