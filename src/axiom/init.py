import jax.numpy as jnp
import jax.nn.initializers as jinit

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
def linspace(start: float, stop: float):
    """
    Creates an initializer that spreads values evenly from start to stop
    across the entire allocated tensor shape.
    """

    def init_fn(key, shape, dtype=jnp.float32):
        # Calculate total parameters and generate the 1D linspace
        num_elements = jnp.prod(jnp.array(shape))
        arr = jnp.linspace(start, stop, num_elements, dtype=dtype)
        # Reshape it to perfectly fit the requested tensor allocation
        return jnp.reshape(arr, shape)

    return init_fn