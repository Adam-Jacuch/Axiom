from axiom.exceptions import AxiomShapeError


class SymbolicSize:
    """Represents a deferred dimension size (e.g., ax.d * 2) to be resolved at runtime."""

    def __init__(self, ref_name: str, op: str, value: int):
        self.ref_name = ref_name
        self.op = op
        self.value = value

    def resolve(self, size_map: dict) -> int:
        if self.ref_name not in size_map:
            raise AxiomShapeError(
                f"Cannot resolve relative size: referenced axis '{self.ref_name}' is not in the current tensor."
            )
        ref_size = size_map[self.ref_name]

        if self.op == "*":
            return ref_size * self.value

        if self.op == "//":
            # The new runtime divisibility guardrail
            if ref_size % self.value != 0:
                raise AxiomShapeError(
                    f"Symbolic floor division failed at runtime: axis '{self.ref_name}' "
                    f"size {ref_size} is not cleanly divisible by {self.value}."
                )
            return ref_size // self.value

        raise ValueError(f"Unknown symbolic op: {self.op}")

    def __repr__(self):
        return f"SymbolicSize({self.ref_name} {self.op} {self.value})"


class ConsumedSlot:
    def __init__(self, name: str, op: str, source_name: str = None):
        self._name = name
        self.op = op
        self.source_name = source_name if source_name is not None else name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"ConsumedSlot('{self._name}', op='{self.op}')"


class CastOp:
    def __init__(self, dtype):
        self.dtype = dtype

    def __repr__(self):
        return f"CastOp({self.dtype.__name__})"


class MaskOp:
    def __init__(self, kind: str, other_axis: str = "left", fill_value=None):
        kind = kind.lower()
        other_axis = other_axis.lower()

        if kind not in ("tril", "triu"):
            raise ValueError("mask() currently supports only 'tril' and 'triu'.")
        if other_axis != "left":
            raise ValueError("mask() currently supports only other_axis='left'.")
        self.kind = kind
        self.other_axis = other_axis
        self.fill_value = fill_value

    def __repr__(self):
        return f"MaskOp(kind='{self.kind}', other_axis='{self.other_axis}')"

class ConvModeOp:
    VALID_MODES = ("same", "valid", "causal")

    def __init__(self, mode: str):
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown conv mode '{mode}'.")
        self.mode = mode

    def __repr__(self):
        return f"ConvModeOp('{self.mode}')"


class ConvStrideOp:
    def __init__(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("stride(...) expects a positive integer.")
        self.value = value

    def __repr__(self):
        return f"ConvStrideOp({self.value})"


class ConvDilationOp:
    def __init__(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("dilate(...) expects a positive integer.")
        self.value = value

    def __repr__(self):
        return f"ConvDilationOp({self.value})"

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
            kernel,
            over=None,
            out_axis=None,
            weight=None,
            bias=None,
            use_bias=True,
            pad=None,
            groups=1,
            kernel_init=None,
            bias_init=None,
            legacy_padding=None,
            legacy_strides=None,
            legacy_dilation=None,
    ):
        if bias is not None and use_bias:
            raise ValueError("conv(...): cannot provide explicit bias and also set use_bias=True.")

        if groups not in (-1, "depthwise"):
            if not isinstance(groups, int) or groups <= 0:
                raise ValueError("conv(...): groups must be a positive integer, -1, or 'depthwise'.")

        self.kernel = kernel
        self.over = over
        self.out_axis = out_axis
        self.weight = weight
        self.bias = bias
        self.use_bias = use_bias
        self.pad = pad
        self.groups = groups
        self.kernel_init = kernel_init
        self.bias_init = bias_init

        # Legacy compatibility path
        self.legacy_padding = legacy_padding
        self.legacy_strides = legacy_strides
        self.legacy_dilation = legacy_dilation

    def __repr__(self):
        out_name = getattr(self.out_axis, "name", None)
        over_name = getattr(self.over, "name", None)
        return (
            f"ConvOp(kernel={self.kernel}, over={over_name}, out={out_name}, "
            f"groups={self.groups}, use_bias={self.use_bias})"
        )


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
    def __init__(self, fn, inputs, init=None):
        self.fn = fn
        self.inputs = inputs
        self.init = init


class AssocScanOp:
    def __init__(self, fn, inputs=None, reverse=False):
        self.fn = fn
        self.inputs = inputs if inputs is not None else tuple()
        self.reverse = reverse

    def __repr__(self):
        return "AssocScanOp()"


class WhereOp:
    def __init__(self, condition, false_tensor):
        self.condition = condition
        self.false_tensor = false_tensor

class PadOp:
    def __init__(self, pad_width, mode="constant", value=0.0):
        self.pad_width = tuple(pad_width)
        self.mode = mode
        self.value = value

class GatherOp:
    def __init__(self, indices):
        self.indices = indices

class RollOp:
    def __init__(self, shift):
        self.shift = shift

class FillOp:
    def __init__(self, value):
        self.value = value

class AttendOp:
    def __init__(self, keys, values, dim, is_causal=False):
        self.keys = keys
        self.values = values
        self.dim = dim
        self.is_causal = is_causal

    def __repr__(self):
        return f"AttendOp(causal={self.is_causal})"

class PowOp:
    def __init__(self, exponent):
        self.exponent = exponent

class ClampOp:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

class StopGradientOp:
    def __repr__(self):
        return "StopGradientOp()"

class ScatterOp:
    def __init__(self, indices, updates, mode="update"):
        if mode not in ("update", "add"):
            raise ValueError("scatter mode must be 'update' or 'add'.")
        self.indices = indices
        self.updates = updates
        self.mode = mode


def _is_pack_safe_axis_op(op) -> bool:
    return isinstance(op, (ConvModeOp, ConvStrideOp, ConvDilationOp))


def _validate_pack_child(axis):
    if not isinstance(axis, Axis):
        raise TypeError("PackedAxis children must be Axis objects.")
    if axis.ops and not all(_is_pack_safe_axis_op(op) for op in axis.ops):
        raise ValueError(
            f"Cannot pack axis '{axis.name}' because it already carries non-packable ops. "
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
            if a.size is None or hasattr(a.size, "resolve"):
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

    def where(self, condition, false_tensor):
        return self._spawn(list(self.ops) + [WhereOp(condition, false_tensor)])  # Use self.__class__(...) for Axis

    def pad(self, pad_width, mode="constant", value=0.0):
        return self._spawn(list(self.ops) + [PadOp(pad_width, mode, value)])

    def gather(self, indices):
        return self._spawn(list(self.ops) + [GatherOp(indices)])

    def roll(self, shift):
        return self._spawn(list(self.ops) + [RollOp(shift)])

    def fill(self, value):
        return self._spawn(list(self.ops) + [FillOp(value)])

    def scan(self, fn, inputs=None, init=None):
        return self._spawn(list(self.ops) + [ScanOp(fn, inputs if inputs is not None else tuple(), init=init)])

    def assoc_scan(self, fn, inputs=None, reverse=False):
        return self._spawn(list(self.ops) + [AssocScanOp(fn, inputs, reverse)])

    def dropout(self, rate=0.1):
        return self._spawn(list(self.ops) + [DropoutOp(rate)])

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

    # --- Phase 1: Pointwise ---
    def clamp(self, min=None, max=None):
        return self._spawn(list(self.ops) + [ClampOp(min_val=min, max_val=max)])

    def exp(self): return self._spawn(list(self.ops) + ["exp"])

    def log(self): return self._spawn(list(self.ops) + ["log"])

    def abs(self): return self._spawn(list(self.ops) + ["abs"])

    def rsqrt(self): return self._spawn(list(self.ops) + ["rsqrt"])

    def swiglu(self): return self._spawn(list(self.ops) + ["swiglu"])

    def square(self):
        return self._spawn(list(self.ops) + ["square"])

    def pow(self, exponent):
        return self._spawn(list(self.ops) + [PowOp(exponent)])

    def round(self):
        return self._spawn(list(self.ops) + ["round"])

    def floor(self):
        return self._spawn(list(self.ops) + ["floor"])

    def ceil(self):
        return self._spawn(list(self.ops) + ["ceil"])

    def sin(self):
        return self._spawn(list(self.ops) + ["sin"])

    def cos(self):
        return self._spawn(list(self.ops) + ["cos"])

    # --- Phase 3: Sparse & Gradient Control ---
    def stop_gradient(self):
        return self._spawn(list(self.ops) + [StopGradientOp()])

    def scatter(self, indices, updates, mode="update"):
        return self._spawn(list(self.ops) + [ScatterOp(indices, updates, mode)])

    def _map_conv_meta(self, method_name, *args, **kwargs):
        if self.ops:
            raise ValueError(
                f"Cannot apply conv-domain modifiers to PackedAxis('{self.name}') "
                f"because it already carries packed ops."
            )
        return PackedAxis(*[
            getattr(a, method_name)(*args, **kwargs) for a in self.axes
        ])

    def same(self):
        return self._map_conv_meta("same")

    def valid(self):
        return self._map_conv_meta("valid")

    def causal(self):
        return self._map_conv_meta("causal")

    def stride(self, n: int):
        return self._map_conv_meta("stride", n)

    def dilate(self, n: int):
        return self._map_conv_meta("dilate", n)

    def __repr__(self):
        return f"PackedAxis({', '.join([a.name for a in self.axes])})"

class Axis:
    # Removed strict int type hint from size to cleanly allow SymbolicSize
    def __init__(self, name: str, size=None, ops: list = None, source_name: str = None):
        self.name = name
        self.size = size
        self.ops = list(ops) if ops is not None else []
        self.source_name = source_name if source_name is not None else name

    def __call__(self, size) -> "Axis":
        # Intercept and map sized dimensions transparently (e.g. ax.d2(ax.d * 2))
        if isinstance(size, Axis):
            size = size.size
        return Axis(self.name, size, list(self.ops), self.source_name)

    def __mul__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError("Axis dimensions must be multiplied by integers.")

        new_size = self.size * scalar if self.size is not None else SymbolicSize(self.name, "*", scalar)
        return Axis(self.name, new_size, list(self.ops), self.source_name)

    def __rmul__(self, scalar: int):
        return self.__mul__(scalar)

    def __floordiv__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError("Axis dimensions must be divided by integers.")

        if self.size is not None:
            if self.size % scalar != 0:
                raise AxiomShapeError(f"Axis '{self.name}' size {self.size} is not cleanly divisible by {scalar}.")
            new_size = self.size // scalar
        else:
            new_size = SymbolicSize(self.name, "//", scalar)

        return Axis(self.name, new_size, list(self.ops), self.source_name)

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
        return self.__class__(self.name, self.size, new_ops, getattr(self, "source_name", None))

    def conv(
            self,
            kernel=None,
            *,
            over=None,
            out=None,
            groups=1,
            weight=None,
            bias=None,
            use_bias=True,
            pad=None,
            kernel_init=None,
            bias_init=None,
            # legacy sugar
            features=None,
            kernel_size=None,
            strides=None,
            padding=None,
            dilation=None,
    ):
        using_legacy = any(v is not None for v in (features, kernel_size, strides, padding, dilation))

        base_self = Axis(self.name, self.size, source_name=self.source_name)

        if using_legacy:
            if kernel is not None or over is not None or out is not None or pad is not None:
                raise ValueError(
                    "conv(...): do not mix legacy arguments "
                    "(features/kernel_size/strides/padding/dilation) with canonical arguments "
                    "(kernel/over/out/pad)."
                )
            if kernel_size is None:
                raise ValueError("conv(...): legacy form requires kernel_size=...")

            out_axis = base_self if features is None else (
                features if isinstance(features, Axis) else Axis(self.name, features)
            )

            op = ConvOp(
                kernel=kernel_size,
                over=None,
                out_axis=out_axis,
                weight=weight,
                bias=bias,
                use_bias=use_bias,
                pad=None,
                groups=groups,
                kernel_init=kernel_init,
                bias_init=bias_init,
                legacy_padding=padding,
                legacy_strides=strides,
                legacy_dilation=dilation,
            )
        else:
            if kernel is None:
                raise ValueError("conv(...): canonical form requires kernel.")
            if over is None:
                raise ValueError("conv(...): canonical form requires over=...")

            out_axis = base_self if out is None else (
                out if isinstance(out, Axis) else Axis(self.name, out)
            )

            op = ConvOp(
                kernel=kernel,
                over=over,
                out_axis=out_axis,
                weight=weight,
                bias=bias,
                use_bias=use_bias,
                pad=pad,
                groups=groups,
                kernel_init=kernel_init,
                bias_init=bias_init,
            )

        return self._spawn(list(self.ops) + [op])

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

    def where(self, condition, false_tensor):
        return Axis(
            self.name, self.size, list(self.ops) + [WhereOp(condition, false_tensor)], self.source_name
        )

    def pad(self, pad_width, mode="constant", value=0.0):
        return Axis(
            self.name, self.size, list(self.ops) + [PadOp(pad_width, mode, value)], self.source_name
        )

    def gather(self, indices):
        return Axis(
            self.name, self.size, list(self.ops) + [GatherOp(indices)], self.source_name
        )

    def roll(self, shift):
        return Axis(
            self.name, self.size, list(self.ops) + [RollOp(shift)], self.source_name
        )

    def fill(self, value):
        return Axis(
            self.name, self.size, list(self.ops) + [FillOp(value)], self.source_name
        )

    def scan(self, fn, inputs=None, init=None):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [ScanOp(fn, inputs if inputs is not None else tuple(), init=init)],
            self.source_name
        )

    def assoc_scan(self, fn, inputs=None, reverse=False):
        return Axis(
            self.name,
            self.size,
            list(self.ops) + [AssocScanOp(fn, inputs, reverse)],
            self.source_name
        )

    def attend(self, keys, values, dim, is_causal=False):
        return Axis(
            self.name, self.size,
            list(self.ops) + [AttendOp(keys, values, dim, is_causal)],
            self.source_name
        )

    def dropout(self, rate=0.1):
        return Axis(self.name, self.size, list(self.ops) + [DropoutOp(rate)], self.source_name)

    def cast(self, dtype):
        return self.__class__(self.name, self.size, list(self.ops) + [CastOp(dtype)],
                              getattr(self, "source_name", None))

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
        return ConsumedSlot(self.name, "sum", source_name=getattr(self, "source_name", self.name))

    def mean(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "mean", source_name=getattr(self, "source_name", self.name))

    def max(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "max", source_name=getattr(self, "source_name", self.name))

    def min(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "min", source_name=getattr(self, "source_name", self.name))

    def var(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "var", source_name=getattr(self, "source_name", self.name))

    def std(self) -> ConsumedSlot:
        return ConsumedSlot(self.name, "std", source_name=getattr(self, "source_name", self.name))

    # --- Phase 1: Pointwise ---
    def clamp(self, min=None, max=None):
        return Axis(self.name, self.size, list(self.ops) + [ClampOp(min_val=min, max_val=max)], self.source_name)

    def exp(self): return Axis(self.name, self.size, list(self.ops) + ["exp"], self.source_name)

    def log(self): return Axis(self.name, self.size, list(self.ops) + ["log"], self.source_name)

    def abs(self): return Axis(self.name, self.size, list(self.ops) + ["abs"], self.source_name)

    def rsqrt(self): return Axis(self.name, self.size, list(self.ops) + ["rsqrt"], self.source_name)

    def swiglu(self): return Axis(self.name, self.size, list(self.ops) + ["swiglu"], self.source_name)

    def square(self):
        return Axis(self.name, self.size, list(self.ops) + ["square"], self.source_name)

    def pow(self, exponent):
        return Axis(self.name, self.size, list(self.ops) + [PowOp(exponent)], self.source_name)

    def round(self):
        return Axis(self.name, self.size, list(self.ops) + ["round"], self.source_name)

    def floor(self):
        return Axis(self.name, self.size, list(self.ops) + ["floor"], self.source_name)

    def ceil(self):
        return Axis(self.name, self.size, list(self.ops) + ["ceil"], self.source_name)

    def sin(self):
        return Axis(self.name, self.size, list(self.ops) + ["sin"], self.source_name)

    def cos(self):
        return Axis(self.name, self.size, list(self.ops) + ["cos"], self.source_name)

    # --- Phase 3: Sparse & Gradient Control ---
    def stop_gradient(self):
        return Axis(self.name, self.size, list(self.ops) + [StopGradientOp()], self.source_name)

    def scatter(self, indices, updates, mode="update"):
        return Axis(self.name, self.size, list(self.ops) + [ScatterOp(indices, updates, mode)], self.source_name)

    # --- Phase 2: Advanced Reductions ---
    def logsumexp(self) -> ConsumedSlot: return ConsumedSlot(self.name, "logsumexp",
                                                             source_name=getattr(self, "source_name", self.name))

    def argmax(self) -> ConsumedSlot: return ConsumedSlot(self.name, "argmax",
                                                          source_name=getattr(self, "source_name", self.name))

    def argmin(self) -> ConsumedSlot: return ConsumedSlot(self.name, "argmin",
                                                          source_name=getattr(self, "source_name", self.name))

    def any(self) -> ConsumedSlot: return ConsumedSlot(self.name, "any",
                                                       source_name=getattr(self, "source_name", self.name))

    def all(self) -> ConsumedSlot: return ConsumedSlot(self.name, "all",
                                                       source_name=getattr(self, "source_name", self.name))

    def _spawn(self, ops=None, size=None):
        return self.__class__(
            self.name,
            self.size if size is None else size,
            list(self.ops if ops is None else ops),
            self.source_name,
        )

    def same(self):
        return self._spawn(list(self.ops) + [ConvModeOp("same")])

    def valid(self):
        return self._spawn(list(self.ops) + [ConvModeOp("valid")])

    def causal(self):
        return self._spawn(list(self.ops) + [ConvModeOp("causal")])

    def stride(self, n: int):
        return self._spawn(list(self.ops) + [ConvStrideOp(n)])

    def dilate(self, n: int):
        return self._spawn(list(self.ops) + [ConvDilationOp(n)])

    def __repr__(self):
        return f"Axis('{self.name}', size={self.size}, ops={self.ops})"