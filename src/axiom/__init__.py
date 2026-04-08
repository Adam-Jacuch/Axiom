from .core.axis import Axis
from .core.tensor import AxiomTensor, vmap
from .core.module import Module, context
from . import init

class _AxisFactory:
    def __getattr__(self, name: str) -> Axis:
        return Axis(name)

# Expose the public API
ax = _AxisFactory()

def tensor(data, *axes) -> AxiomTensor:
    """Helper to wrap a raw JAX array into an AxiomTensor."""
    return AxiomTensor(data, axes)

__all__ = ["ax", "tensor", "Module", "context", "init"]