class ConsumedSlot:
    def __init__(self, source_name: str, op: str):
        self.source_name = source_name
        self.op = op

    @property
    def name(self): return self.source_name

    def __repr__(self): return f"ConsumedSlot('{self.source_name}', op='{self.op}')"


class CastOp:
    def __init__(self, dtype):
        self.dtype = dtype
    def __repr__(self): return f"CastOp({self.dtype.__name__})"


class ProjOp:
    def __init__(self, out_axis, weight=None, use_bias=False, kernel_init=None, bias_init=None, dtype=None):
        self.out_axis = out_axis
        self.weight = weight
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dtype = dtype

    def __repr__(self): return f"ProjOp(out={self.out_axis.name})"


class ConvOp:
    def __init__(self, features, kernel_size, strides=None, padding='SAME', use_bias=True, kernel_init=None, bias_init=None):
        self.features = features
        self.kernel_size = kernel_size
        self.strides = strides if strides else (1,) * len(kernel_size)
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
    def __repr__(self): return f"ConvOp(kernel_size={self.kernel_size})"


class BiasOp:
    def __init__(self, init_fn=None):
        self.init_fn = init_fn

    def __repr__(self): return "BiasOp()"


class GateOp:
    def __init__(self, gate_tensor):
        self.gate_tensor = gate_tensor

    def __repr__(self): return "GateOp()"


class NormOp:
    def __init__(self, norm_type, eps=1e-5, use_bias=True, use_scale=True, init_scale=None, init_bias=None):
        self.norm_type = norm_type
        self.eps = eps
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.init_scale = init_scale
        self.init_bias = init_bias

    def __repr__(self): return f"NormOp('{self.norm_type}')"


class DropoutOp:
    def __init__(self, rate):
        self.rate = rate

    def __repr__(self): return f"DropoutOp({self.rate})"


class ScanOp:
    def __init__(self, fn, inputs):
        self.fn = fn
        self.inputs = inputs

    def __repr__(self): return f"ScanOp(fn={self.fn.__name__})"


class PackedAxis:
    def __init__(self, *axes, ops=None):
        self.axes = axes
        self.ops = ops or []

    def __and__(self, other):
        if isinstance(other, PackedAxis):
            return PackedAxis(*self.axes, *other.axes, ops=self.ops)
        elif isinstance(other, 'Axis'):
            return PackedAxis(*self.axes, other, ops=self.ops)
        raise TypeError("Can only pack Axis or PackedAxis.")

    @property
    def name(self):
        return "&".join([a.name for a in self.axes])

    @property
    def size(self):
        total = 1
        for a in self.axes:
            if a.size is None: return None
            total *= a.size
        return total

    # --- Standard Lib ---
    def proj(self, out=None, weight=None, use_bias=False, kernel_init=None, bias_init=None, dtype=None):
        out_axis = out if out is not None else self
        import copy
        new_packed = copy.copy(self) # Safely duplicates without triggering __init__
        new_packed.ops = self.ops + [ProjOp(out_axis, weight, use_bias, kernel_init, bias_init, dtype)]
        return new_packed

    def bias(self, init_fn=None):
        return PackedAxis(*self.axes, ops=self.ops + [BiasOp(init_fn)])

    def gate(self, tensor):
        return PackedAxis(*self.axes, ops=self.ops + [GateOp(tensor)])

    def norm_rms(self, eps=1e-5, init_scale=None):
        return PackedAxis(*self.axes, ops=self.ops + [NormOp('rms', eps, False, True, init_scale, None)])

    def norm_layer(self, eps=1e-5, use_bias=True, use_scale=True):
        return PackedAxis(*self.axes, ops=self.ops + [NormOp('layer', eps, use_bias, use_scale)])

    def dropout(self, rate=0.1):
        return PackedAxis(*self.axes, ops=self.ops + [DropoutOp(rate)])

    def scan(self, fn, inputs=None):
        return PackedAxis(*self.axes, ops=self.ops + [ScanOp(fn, inputs if inputs is not None else tuple())])

    def cast(self, dtype):
        return self.__class__(self.name, self.size, self.ops + [CastOp(dtype)], getattr(self, 'source_name', None))

    # --- Activations ---
    def relu(self):
        return PackedAxis(*self.axes, ops=self.ops + ['relu'])

    def silu(self):
        return PackedAxis(*self.axes, ops=self.ops + ['silu'])

    def gelu(self):
        return PackedAxis(*self.axes, ops=self.ops + ['gelu'])

    def sigmoid(self):
        return PackedAxis(*self.axes, ops=self.ops + ['sigmoid'])

    def tanh(self):
        return PackedAxis(*self.axes, ops=self.ops + ['tanh'])

    def softmax(self):
        return PackedAxis(*self.axes, ops=self.ops + ['softmax'])

    def __repr__(self):
        return f"PackedAxis({', '.join([a.name for a in self.axes])})"


class Axis:
    def __init__(self, name: str, size: int = None, ops: list = None, source_name: str = None):
        self.name = name
        self.size = size
        self.ops = ops or []
        self.source_name = source_name if source_name is not None else name

    def __call__(self, size: int) -> 'Axis':
        return Axis(self.name, size, list(self.ops), self.source_name)

    def __and__(self, other):
        if isinstance(other, PackedAxis):
            return PackedAxis(self, *other.axes)
        elif isinstance(other, Axis):
            return PackedAxis(self, other)
        raise TypeError("Can only pack Axis or PackedAxis.")

    def __rshift__(self, other):
        """Enables in-flight renaming: ax.sq >> ax.sk"""
        new_name = other.name if hasattr(other, 'name') else str(other)
        # Preserve the original name as the 'source_name' so the parser can locate it!
        source = getattr(self, 'source_name', self.name)
        return self.__class__(new_name, self.size, self.ops, source_name=source)

    # --- Standard Lib ---
    def proj(self, out=None, weight=None, use_bias=False, kernel_init=None, bias_init=None, dtype=None):
        out_axis = out if out is not None else self
        new_ops = self.ops + [ProjOp(out_axis, weight, use_bias, kernel_init, bias_init, dtype)]
        return self.__class__(out_axis.name, out_axis.size, new_ops, getattr(self, 'source_name', None))

    def conv(self, features, kernel_size, strides=None, padding='SAME', use_bias=True, kernel_init=None,
             bias_init=None):
        out_axis = features if isinstance(features, Axis) else Axis(self.name, features)
        new_ops = self.ops + [ConvOp(out_axis, kernel_size, strides, padding, use_bias, kernel_init, bias_init)]
        return Axis(out_axis.name, out_axis.size, new_ops, self.source_name)

    def bias(self, init_fn=None):
        return Axis(self.name, self.size, self.ops + [BiasOp(init_fn)], self.source_name)

    def gate(self, tensor):
        return Axis(self.name, self.size, self.ops + [GateOp(tensor)], self.source_name)

    def mask(self, mask_type: str):
        return Axis(self.name, self.size, self.ops + [f'mask_{mask_type}'], self.source_name)

    def norm_rms(self, eps=1e-5, init_scale=None):
        return Axis(self.name, self.size, self.ops + [NormOp('rms', eps, False, True, init_scale, None)],
                    self.source_name)

    def norm_layer(self, eps=1e-5, use_bias=True, use_scale=True):
        return Axis(self.name, self.size, self.ops + [NormOp('layer', eps, use_bias, use_scale)], self.source_name)

    def dropout(self, rate=0.1):
        return Axis(self.name, self.size, self.ops + [DropoutOp(rate)], self.source_name)

    def scan(self, fn, inputs=None):
        return Axis(self.name, self.size, self.ops + [ScanOp(fn, inputs if inputs is not None else tuple())],
                    self.source_name)

    def cast(self, dtype):
        return self.__class__(self.name, self.size, self.ops + [CastOp(dtype)], getattr(self, 'source_name', None))

    # --- Activations ---
    def relu(self):
        return Axis(self.name, self.size, self.ops + ['relu'], self.source_name)

    def silu(self):
        return Axis(self.name, self.size, self.ops + ['silu'], self.source_name)

    def gelu(self):
        return Axis(self.name, self.size, self.ops + ['gelu'], self.source_name)

    def sigmoid(self):
        return Axis(self.name, self.size, self.ops + ['sigmoid'], self.source_name)

    def tanh(self):
        return Axis(self.name, self.size, self.ops + ['tanh'], self.source_name)

    def softmax(self):
        return Axis(self.name, self.size, self.ops + ['softmax'], self.source_name)

    # --- Reductions ---
    def sum(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'sum')

    def mean(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'mean')

    def max(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'max')

    def min(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'min')

    def var(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'var')

    def std(self) -> ConsumedSlot:
        return ConsumedSlot(self.source_name, 'std')

    def __repr__(self):
        return f"Axis('{self.name}', size={self.size}, ops={self.ops})"