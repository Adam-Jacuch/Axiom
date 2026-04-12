import jax.numpy as jnp
import jax.nn.initializers as jinit

import numpy as np

# Standard constant initializers
zeros = jinit.zeros
ones = jinit.ones
constant = jinit.constant

# Variance scaling initializers
uniform = jinit.uniform
normal = jinit.normal

# ML Standard Initializers
xavier_uniform = jinit.glorot_uniform
xavier_normal = jinit.glorot_normal
kaiming_uniform = jinit.he_uniform
kaiming_normal = jinit.he_normal
orthogonal = jinit.orthogonal

# Axiom Defaults
default_kernel_init = jinit.lecun_normal()
default_bias_init = jinit.zeros


# ==========================================
# Custom Axiom Initializers
# ==========================================
def linspace(start, stop, dtype=jnp.float32):
    def init_fn(key, shape):
        num_elements = shape[0]

        # Trillium Guardrail for Tracers
        if hasattr(num_elements, "val"):
            num = int(num_elements.val)
        else:
            num = int(num_elements)

        return jnp.array(np.linspace(start, stop, num, dtype=dtype))

    return init_fn