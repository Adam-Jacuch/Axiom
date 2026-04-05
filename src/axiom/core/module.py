import functools
import threading
from flax import nnx


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

    def get_rngs(self):
        # 1. Check if the root Model has an explicit 'rngs' attached
        if self.stack and hasattr(self.stack[0], 'rngs'):
            return self.stack[0].rngs
        # 2. Fallback to an implicit shared global RNG
        if not hasattr(self._local, "rngs"):
            self._local.rngs = nnx.Rngs(params=0, dropout=1)
        return self._local.rngs


context = _ModuleContext()


class Module(nnx.Module):
    """Base class for Axiom models. Injects context tracking into __call__."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            original_call = cls.__dict__["__call__"]

            @functools.wraps(original_call)
            def wrapped_call(self, *args, **kwds):
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

            cls.__call__ = wrapped_call