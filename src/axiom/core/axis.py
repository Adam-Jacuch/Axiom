from axiom.exceptions import AxiomShapeError


class ConsumedSlot:
    def __init__(self, source_name: str, op: str):
        self.source_name = source_name
        self.op = op

    @property
    def name(self):
        return self.source_name

    def __repr__(self):
        return f"ConsumedSlot('{self.source_name}', op='{self.op}')"


class CastOp:
    def __init__(self, dtype):
        self.dtype = dtype

    def __repr__(self):
        return f"CastOp({self.dtype.__name__})"


class MaskOp:
    def __init__(self, kind: str, other_axis: str = "left", fill_value=None):
        if kind not in ("tril", "triu"):
            raise ValueError("mask() currently supports only 'tril' and 'triu'.")
        if other_axis != "left":
            raise ValueError("mask() currently supports only other_axis='left'.")
        self.kind = kind
        self.other_axis = other_axis
        self.fill_value = fill_value

    def __repr__(self):
        return f"MaskOp(kind='{self.kind}', other_axis='{self.other_axis}')"


class ProjOp:
    def __init__(
        self,
        out_axis,
        weight=None,
        bias=None,
        use_bias=True,
        kernel_init=None,
        bias_init=None,
        dtype=None,
    ):
        if bias is not None and use_bias:
            raise ValueError("proj(...): cannot provide explicit bias and also set use_bias=True.")
        self.out_axis = out_axis
        self.weight = weight
        self.bias = bias
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dtype = dtype

    def __repr__(self):
        return f"ProjOp(out={self.out_axis.name}, use_bias={self.use_bias})"


class ConvOp:
    def __init__(
        self,
        features,
        kernel_size,
        strides=None,
        padding="SAME",
        use_bias=True,
        kernel_init=None,
        bias_init=None,
        weight=None,
        bias=None,
    ):
        if bias is not None and use_bias:
            raise ValueError("conv(...): cannot provide explicit bias and also set use_bias=True.")
        self.features = features
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides) if strides is not None else (1,) * len(self.kernel_size)
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.weight = weight
        self.bias = bias

    def __repr__(self):
        return f"ConvOp(kernel_size={self.kernel_size}, use_bias={self.use_bias})"


class BiasOp:
    def __init__(self, tensor=None, init_fn=None):
        self.tensor = tensor
        self.init_fn = init_fn

    def __repr__(self):
        return "BiasOp()"


class GateOp:
    def __init__(self, tensor=None, init_fn=None):
        self.tensor = tensor
        self.init_fn = init_fn

    def __repr__(self):
        return "GateOp()"


class NormOp:
    def __init__(
        self,
        norm_type,
        eps=1e-5,
        use_bias=True,
        use_scale=True,
        init_scale=None,
        init_bias=None,
        scale=None,
        bias=None,
    ):
        if norm_type not in ("rms", "layer"):
            raise ValueError("norm_type must be 'rms' or 'layer'.")

        if norm_type == "rms":
            if bias is not None or use_bias:
                raise ValueError("norm_rms() does not support bias.")
        else:
            if bias is not None and use_bias:
                raise ValueError("norm_layer(...): cannot provide explicit bias and also set use_bias=True.")

        if scale is not None and use_scale:
            raise ValueError("norm(...): cannot provide explicit scale and also set use_scale=True.")

        self.norm_type = norm_type
        self.eps = eps
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.init_scale = init_scale
        self.init_bias = init_bias
        self.scale = scale
        self.bias = bias

    def __repr__(self):
        return f"NormOp('{self.norm_type}')"


class DropoutOp:
    def __init__(self, rate):
        self.rate = rate

    def __repr__(self):
        return f"DropoutOp({self.rate})"


class ScanOp:
    def __init__(self, fn, inputs):
        self.fn = fn
        self.inputs = inputs

    def __repr__(self):
        return f"ScanOp(fn={self.fn.__name__})"


def _validate_pack_child(axis):
    if not isinstance(axis, Axis):
        raise TypeError("PackedAxis children must be Axis objects.")
    if axis.ops:
        raise ValueError(
            f"Cannot pack axis '{axis.name}' because it already carries ops. "
            f"Apply ops before or after packing, not on child axes inside a pack."
        )


def _validate_pack_operand(obj):
    if isinstance(obj, PackedAxis):
        if obj.ops:
            raise ValueError(
                f"Cannot further pack PackedAxis('{obj.name}') because it already carries packed ops."
            )
        for a in obj.axes:
            _validate_pack_child(a)
    elif isinstance(obj, Axis):
        _validate_pack_child(obj)
    else:
        raise TypeError("Can only pack Axis or PackedAxis.")


class PackedAxis:
    def __init__(self, *axes, ops=None):
        for a in axes:
            _validate_pack_child(a)
        self.axes = tuple(axes)
        self.ops = list(ops) if ops is not None else []

    def _spawn(self, ops=None):
        new = PackedAxis(*self.axes, ops=list(self.ops if ops is None else ops))
        if hasattr(self, "source_name"):
            new.source_name = self.source_name
        return new

    def __and__(self, other):
        _validate_pack_operand(self)
        _validate_pack_operand(other)

        if isinstance(other, PackedAxis):
            return PackedAxis(*self.axes, *other.axes, ops=list(self.ops))
        if isinstance(other, Axis):
            return PackedAxis(*self.axes, other, ops=list(self.ops))
        raise TypeError("Can only pack Axis or PackedAxis.")

    def __rshift__(self, other):
        import copy

        if hasattr(other, "axes"):
            new_axis = copy.copy(other)
            new_axis.source_name = getattr(self, "source_name", self.name)
            return new_axis

        new_name = other.name if hasattr(other, "name") else str(other)
        source = getattr(self, "source_name", self.name)
        return other.__class__(
            new_name,
            getattr(other, "size", None),
            list(getattr(other, "ops", [])),
            source_name=source,
        )

    @property
    def name(self):
        return "&".join([a.name for a in self.axes])

    @property
    def size(self):
        total = 1
        for a in self.axes:
            if a.size is None:
                return None
            total *= a.size
        return total

    def proj(
        self,
        out=None,
        weight=None,
        bias=None,
        use_bias=True,
        kernel_init=None,
        bias_init=None,
        dtype=None,
    ):
        out_axis = out if out is not None else self
        return self._spawn(
            list(self.ops) + [
                ProjOp(
                    out_axis,
                    weight=weight,
                    bias=bias,
                    use_bias=use_bias,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    dtype=dtype,
                )
            ]
        )

    def bias(self, tensor=None, init_fn=None):
        return self._spawn(list(self.ops) + [BiasOp(tensor=tensor, init_fn=init_fn)])

    def gate(self, tensor=None, init_fn=None):
        return self._spawn(list(self.ops) + [GateOp(tensor=tensor, init_fn=init_fn)])

    def norm_rms(self, eps=1e-5, scale=None, init_scale=None):
        return self._spawn(
            list(self.ops) + [
                NormOp(
                    "rms",
                    eps,
                    use_bias=False,
                    use_scale=(scale is None),
                    init_scale=init_scale,
                    init_bias=None,
                    scale=scale,
                    bias=None,
                )
            ]
        )

    def norm_layer(
        self,
        eps=1e-5,
        use_bias=True,
        use_scale=True,
        scale=None,
        bias=None,
        init_scale=None,
        init_bias=None,
    ):
        return self._spawn(
            list(self.ops) + [
                NormOp(
                    "layer",
                    eps,
                    use_bias=use_bias,
                    use_scale=use_scale,
                    init_scale=init_scale,
                    init_bias=init_bias,
                    scale=scale,
                    bias=bias,
                )
            ]
        )

    def dropout(self, rate=0.1):
        return self._spawn(list(self.ops) + [DropoutOp(rate)])

    def scan(self, fn, inputs=None):
        return self._spawn(list(self.ops) + [ScanOp(fn, inputs if inputs is not None else tuple())])

    def cast(self, dtype):
        return self._spawn(list(self.ops) + [CastOp(dtype)])

    def relu(self):
        return self._spawn(list(self.ops) + ["relu"])

    def silu(self):
        return self._spawn(list(self.ops) + ["silu"])

    def gelu(self):
        return self._spawn(list(self.ops) + ["gelu"])

    def sigmoid(self):
        return self._spawn(list(self.ops) + ["sigmoid"])

    def tanh(self):
        return self._spawn(list(self.ops) + ["tanh"])

    def softmax(self):
        return self._spawn(list(self.ops) + ["softmax"])

    def __repr__(self):
        return f"PackedAxis({', '.join([a.name for a in self.axes])})"


class Axis:
    def __init__(self, name: str, size: int = None, ops: list = None, source_name: str = None):
        self.name = name
        self.size = size
        self.ops = list(ops) if ops is not None else []
        self.source_name = source_name if source_name is not None else name

    def __call__(self, size: int) -> "Axis":
        return Axis(self.name, size, list(self.ops), self.source_name)

    def __mul__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError("Axis dimensions must be multiplied by integers.")
        if self.size is None:
            raise AxiomShapeError(f"Cannot multiply axis '{self.name}' because its physical size is unknown.")

        # Returns a new Axis with the scaled size, preserving ops and source identity
        return Axis(self.name, self.size * scalar, list(self.ops), self.source_name)

    def __rmul__(self, scalar: int):
        return self.__mul__(scalar)

    def __floordiv__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError("Axis dimensions must be divided by integers.")
        if self.size is None:
            raise AxiomShapeError(f"Cannot divide axis '{self.name}' because its physical size is unknown.")
        if self.size % scalar != 0:
            raise AxiomShapeError(f"Axis '{self.name}' size {self.size} is not cleanly divisible by {scalar}.")

        return Axis(self.name, self.size // scalar, list(self.ops), self.source_name)

    def __and__(self, other):
        _validate_pack_operand(self)
        _validate_pack_operand(other)

        if isinstance(other, PackedAxis):
            return PackedAxis(self, *other.axes)
        if isinstance(other, Axis):
            return PackedAxis(self, other)
        raise TypeError("Can only pack Axis or PackedAxis.")

    def __rshift__(self, other):
        import copy

        if hasattr(other, "axes"):
            new_packed = copy.copy(other)
            new_packed.source_name = getattr(self, "source_name", self.name)
            return new_packed

        new_name = other.name if hasattr(other, "name") else str(other)
        source = getattr(self, "source_name", self.name)
        return self.__class__(new_name, self.size, list(self.ops), source_name=source)

    def proj(
        self,
        out=None,
        weight=None,
        bias=None,
        use_bias=True,
        kernel_init=None,
        bias_init=None,
        dtype=None,
    ):
        out_axis = out if out is not None else self
        new_ops = list(self.ops) + [
            ProjOp(
                out_axis,
                weight=weight,
                bias=bias,
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                dtype=dtype,
            )
        ]
        # Keep the current axis identity until the projection actually executes.
        return self.__class__(self.name, self.size, new_ops, getattr(self, "source_name", None))

    def conv(
        self,
        features,
        kernel_size,
        strides=None,
        padding="SAME",
        use_bias=True,
        kernel_init=None,
        bias_init=None,
        weight=None,
        bias=None,
    ):
        out_axis = features if isinstance(features, Axis) else Axis(self.name, features)
        new_ops = list(self.ops) + [
            ConvOp(
                out_axis,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                weight=weight,
                bias=bias,
            )
        ]
        # Keep the current axis identity until the convolution actually executes.
        return Axis(self.name, self.size, new_ops, self.source_name)

    def bias(self, tensor=None, init_fn=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [BiasOp(tensor=tensor, init_fn=init_fn)],
            self.source_name,
        )

    def gate(self, tensor=None, init_fn=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [GateOp(tensor=tensor, init_fn=init_fn)],
            self.source_name,
        )

    def mask(self, kind: str, other_axis: str = "left", fill_value=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [MaskOp(kind=kind, other_axis=other_axis, fill_value=fill_value)],
            self.source_name,
        )

    def norm_rms(self, eps=1e-5, scale=None, init_scale=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops)
            + [
                NormOp(
                    "rms",
                    eps,
                    use_bias=False,
                    use_scale=(scale is None),
                    init_scale=init_scale,
                    init_bias=None,
                    scale=scale,
                    bias=None,
                )
            ],
            self.source_name,
        )

    def norm_layer(
        self,
        eps=1e-5,
        use_bias=True,
        use_scale=True,
        scale=None,
        bias=None,
        init_scale=None,
        init_bias=None,
    ):
        return Axis(
            self.name,
            self.size,
            list(self.ops)
            + [
                NormOp(
                    "layer",
                    eps,
                    use_bias=use_bias,
                    use_scale=use_scale,
                    init_scale=init_scale,
                    init_bias=init_bias,
                    scale=scale,
                    bias=bias,
                )
            ],
            self.source_name,
        )

    def dropout(self, rate=0.1):
        return Axis(self.name, self.size, list(self.ops) + [DropoutOp(rate)], self.source_name)

    def scan(self, fn, inputs=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [ScanOp(fn, inputs if inputs is not None else tuple())],
            self.source_name,
        )

    def cast(self, dtype):
        return self.__class__(self.name, self.size, list(self.ops) + [CastOp(dtype)], getattr(self, "source_name", None))

    def relu(self):
        return Axis(self.name, self.size, list(self.ops) + ["relu"], self.source_name)

    def silu(self):
        return Axis(self.name, self.size, list(self.ops) + ["silu"], self.source_name)

    def gelu(self):
        return Axis(self.name, self.size, list(self.ops) + ["gelu"], self.source_name)

    def sigmoid(self):
        return Axis(self.name, self.size, list(self.ops) + ["sigmoid"], self.source_name)

    def tanh(self):
        return Axis(self.name, self.size, list(self.ops) + ["tanh"], self.source_name)

    def softmax(self):
        return Axis(self.name, self.size, list(self.ops) + ["softmax"], self.source_name)

    def sum(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "sum")

    def mean(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "mean")

    def max(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "max")

    def min(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "min")

    def var(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "var")

    def std(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "std")

    def __repr__(self):
        return f"Axis('{self.name}', size={self.size}, ops={self.ops})"