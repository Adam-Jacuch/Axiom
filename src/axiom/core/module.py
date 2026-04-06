import functools
import threading
from flax import nnx
import jax
import jax.numpy as jnp


class _ModuleContext:
    def __init__(self):
        self._local = threading.local()

    @property
    def stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def get_active(self):
        return self.stack[-1] if self.stack else None

    # The pure functional key property
    @property
    def step_key(self):
        return getattr(self._local, "step_key", None)

    @step_key.setter
    def step_key(self, value):
        self._local.step_key = value


context = _ModuleContext()


class Module(nnx.Module):
    """Base class for Axiom models. Injects context tracking into __call__."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            original_call = cls.__dict__["__call__"]

            @functools.wraps(original_call)
            def wrapped_call(self, *args, **kwds):
                is_root = len(context.stack) == 0

                if is_root:
                    # 1. Initialize the step counter on the very first root call
                    if not hasattr(self, "_axiom_step"):
                        self._axiom_step = nnx.Variable(jnp.array(0))
                        self._axiom_base_key = nnx.Variable(jax.random.key(id(self)))

                    # 2. Generate a pure random key outside the remat boundary
                    current_key = jax.random.fold_in(self._axiom_base_key[...], self._axiom_step[...])

                    # 3. Mutate state safely (remat cannot see this)
                    self._axiom_step[...] += 1

                    # 4. Store it for the dropouts to read
                    context.step_key = current_key

                context.stack.append(self)
                object.__setattr__(self, '_axiom_param_counter', 0)
                try:
                    result = original_call(self, *args, **kwds)
                except Exception:
                    raise
                else:
                    object.__setattr__(self, '_axiom_initialized', True)
                    return result
                finally:
                    context.stack.pop()
                    if is_root:
                        context.step_key = None  # Clean up after the forward pass

            cls.__call__ = wrapped_call