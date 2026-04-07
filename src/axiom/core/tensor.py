import math
import numbers
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx

from .axis import (
    Axis, BiasOp, CastOp, ConsumedSlot, ConvOp, DropoutOp, GateOp, MaskOp,
    NormOp, PackedAxis, ProjOp, ScanOp, SymbolicSize, WhereOp, PadOp, GatherOp,
    RollOp, FillOp, AttendOp,
    ClampOp, StopGradientOp, ScatterOp, AssocScanOp,
    ConvModeOp, ConvStrideOp, ConvDilationOp, PowOp,
)
from .module import context
from .. import init as a_init
from ..exceptions import AxiomShapeError, AxiomSyntaxError


class _AxiomIndexer:
    def __init__(self, owner: "AxiomTensor"):
        self._owner = owner

    def __getitem__(self, item):
        return self._owner._positional_index(item)


@jax.tree_util.register_pytree_node_class
class AxiomTensor:
    def __init__(self, data: jnp.ndarray, logical_axes: Tuple[Axis, ...]):
        self.data = data
        self.axes = logical_axes
        self._assert_unique_names([a.name for a in self.axes], "AxiomTensor initialization")

    def tree_flatten(self):
        axes_spec = tuple((a.name, a.size, a.source_name) for a in self.axes)
        return (self.data,), axes_spec

    @classmethod
    def tree_unflatten(cls, axes_spec, children):
        (data,) = children
        axes = tuple(Axis(name, size, source_name=source_name) for name, size, source_name in axes_spec)
        return cls(data, axes)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def rename(self, old_axis, new_axis):
        new_axes = tuple(new_axis if a.name == old_axis.name else a for a in self.axes)
        return AxiomTensor(self.data, new_axes)

    def _resolve_token_sizes(self, token, size_map: dict):
        """Recursively evaluates SymbolicSize promises into concrete integers."""
        if isinstance(token, Axis):
            if hasattr(token.size, "resolve"):
                size = token.size.resolve(size_map)
            # --- ADD THESE TWO LINES ---
            elif token.size is None and token.name in size_map:
                size = size_map[token.name]
            # ---------------------------
            else:
                size = token.size
            return Axis(token.name, size, list(token.ops), getattr(token, "source_name", token.name))

        elif isinstance(token, PackedAxis):
            new_axes = []
            for a in token.axes:
                if hasattr(a.size, "resolve"):
                    size = a.size.resolve(size_map)
                elif a.size is None and a.name in size_map:
                    size = size_map[a.name]
                else:
                    size = a.size
                new_axes.append(Axis(a.name, size, list(a.ops), getattr(a, "source_name", a.name)))

            new_pack = PackedAxis(*new_axes, ops=list(token.ops))
            if hasattr(token, "source_name"):
                new_pack.source_name = token.source_name
            return new_pack

        return token

    def embed(self, out: Axis = None, vocab: int = None, weight=None, dtype=None, return_weight=False):
        size_map = {a.name: int(self.data.shape[i]) for i, a in enumerate(self.axes)}
        if out is not None:
            out = self._resolve_token_sizes(out, size_map)
        if hasattr(vocab, "resolve"):
            vocab = vocab.resolve(size_map)

        if weight is not None:
            if not isinstance(weight, AxiomTensor):
                raise TypeError("Explicit embed(weight=...) expects an AxiomTensor embedding table.")
            if weight.data.ndim != 2 or len(weight.axes) != 2:
                raise AxiomShapeError("Explicit embedding weights must be rank-2 with axes (vocab, out).")

            vocab_axis, out_axis = weight.axes
            out_axis = Axis(out_axis.name, out_axis.size or weight.data.shape[1], source_name=out_axis.name)

            if vocab is not None:
                known_vocab = vocab_axis.size or weight.data.shape[0]
                if known_vocab != vocab:
                    raise AxiomShapeError(
                        f"Explicit embedding weight vocab size mismatch: expected {vocab}, got {known_vocab}."
                    )

            if out is not None:
                if out.name != out_axis.name:
                    raise AxiomShapeError(
                        f"Explicit embedding output axis mismatch: expected '{out.name}', got '{out_axis.name}'."
                    )
                if out.size is not None and out_axis.size is not None and out.size != out_axis.size:
                    raise AxiomShapeError(
                        f"Explicit embedding output size mismatch: expected {out.size}, got {out_axis.size}."
                    )
                out_axis = Axis(out.name, out.size or out_axis.size, source_name=out.name)

            new_data = jnp.take(weight.data, self.data, axis=0)
            out_tensor = AxiomTensor(new_data, (*self.axes, out_axis))
            return (out_tensor, weight) if return_weight else out_tensor

        # --- Implicit Embedding Logic ---
        active_mod = context.get_active()
        if active_mod is None:
            raise RuntimeError("Implicit .embed() must be called inside an axiom.Module.")

        if isinstance(vocab, Axis):
            vocab_name = vocab.name
            vocab_size = vocab.size
        else:
            # Fallback if someone just passes vocab=256
            vocab_name = "vocab"
            vocab_size = vocab

        if vocab_size is None or out is None:
            raise ValueError("Implicit .embed() requires both a sized vocab axis and an out axis.")
        if out.size is None:
            raise ValueError("Embedding requires an explicit output size, e.g. ax.d(128).")

        param_name = f"_axiom_embed_{active_mod._axiom_param_counter}"
        object.__setattr__(active_mod, '_axiom_param_counter', active_mod._axiom_param_counter + 1)

        if not getattr(active_mod, "_axiom_initialized", False):
            seed = id(active_mod) + active_mod._axiom_param_counter
            e_dtype = dtype if dtype is not None else jnp.float32
            setattr(
                active_mod,
                param_name,
                nnx.Embed(
                    num_embeddings=vocab_size,
                    features=out.size,
                    dtype=e_dtype,
                    param_dtype=e_dtype,
                    rngs=nnx.Rngs(params=seed),  # Private RNG
                ),
            )

        embed_layer = getattr(active_mod, param_name)
        new_data = embed_layer(self.data)
        out_axis = Axis(out.name, out.size, source_name=out.name)
        out_tensor = AxiomTensor(new_data, (*self.axes, out_axis))

        if return_weight:
            w_data = embed_layer.embedding.get_value()
            w_tensor = AxiomTensor(w_data, (Axis(vocab_name, vocab_size), Axis(out.name, out.size)))
            return out_tensor, w_tensor

        return out_tensor

    @property
    def idx(self):
        return _AxiomIndexer(self)

    def _is_int_index(self, token) -> bool:
        return isinstance(token, numbers.Integral) and not isinstance(token, bool)

    def _normalize_positional_index(self, item):
        """
        Normalize positional indexing into a full-rank tuple of only:
        - int
        - slice
        - Ellipsis (expanded away)

        Rules:
        - Only one Ellipsis is allowed
        - None / newaxis is not supported
        - Fancy indexing is not supported
        - If fewer indices than rank are provided, append trailing full slices
        """
        if not isinstance(item, tuple):
            item = (item,)

        if item.count(Ellipsis) > 1:
            raise AxiomSyntaxError("idx[...] allows at most one ellipsis (...).")

        for token in item:
            if token is Ellipsis:
                continue
            if token is None:
                raise AxiomShapeError("idx[...] does not support None / newaxis.")
            if self._is_int_index(token):
                continue
            if isinstance(token, slice):
                continue
            raise AxiomShapeError(
                f"idx[...] only supports integers, slices, and ellipsis. Got {type(token).__name__}."
            )

        specified = sum(token is not Ellipsis for token in item)
        if specified > len(self.axes):
            raise AxiomShapeError(
                f"Too many indices for tensor rank {len(self.axes)}: got {specified} positional indices."
            )

        if Ellipsis in item:
            ellipsis_pos = item.index(Ellipsis)
            fill = len(self.axes) - specified
            item = item[:ellipsis_pos] + (slice(None),) * fill + item[ellipsis_pos + 1:]
        else:
            item = item + (slice(None),) * (len(self.axes) - specified)

        return item

    def _positional_index(self, item) -> "AxiomTensor":
        """
        Pure positional indexing escape hatch.

        Examples:
            x.idx[0]
            x.idx[:, 2:8, :]
            x.idx[..., 1]

        Semantics:
        - int drops the indexed axis
        - slice keeps the axis and updates its size from the resulting tensor
        - ellipsis expands to full slices
        """
        item = self._normalize_positional_index(item)

        new_data = self.data[item]

        new_axes = []
        out_dim = 0
        axis_idx = 0

        for token in item:
            axis = self.axes[axis_idx]

            if self._is_int_index(token):
                # JAX drops this physical dimension entirely.
                axis_idx += 1
                continue

            if isinstance(token, slice):
                new_axes.append(
                    Axis(
                        axis.name,
                        int(new_data.shape[out_dim]),
                        source_name=axis.source_name,
                    )
                )
                out_dim += 1
                axis_idx += 1
                continue

            # Should be unreachable because normalization validates everything.
            raise AxiomShapeError(f"Unsupported idx token: {token!r}")

        return AxiomTensor(new_data, tuple(new_axes))

    # -------------------------------------------------------------------------
    # Safety helpers
    # -------------------------------------------------------------------------

    def _assert_unique_names(self, names, where: str):
        seen = set()
        dups = []
        for n in names:
            if n in seen and n not in dups:
                dups.append(n)
            seen.add(n)
        if dups:
            raise AxiomShapeError(f"Duplicate logical axis names {dups} are not allowed ({where}).")

    def _ordered_unique(self, names):
        out = []
        for name in names:
            if name not in out:
                out.append(name)
        return out

    def _require_active_module(self, op_name: str):
        active_mod = context.get_active()
        if active_mod is None:
            raise RuntimeError(f"{op_name} must be inside an axiom.Module.")
        return active_mod

    # -------------------------------------------------------------------------
    # Arithmetic (raw JAX broadcasting intentionally allowed)
    # -------------------------------------------------------------------------

    def _binary_op(self, other, fn, op_name: str):
        if isinstance(other, AxiomTensor):
            self_names = [a.name for a in self.axes]
            other_names = [a.name for a in other.axes]
            if self_names != other_names:
                raise AxiomShapeError(f"Cannot {op_name} tensors with mismatched axes: {self_names} vs {other_names}")
            return AxiomTensor(fn(self.data, other.data), self.axes)

        return AxiomTensor(fn(self.data, other), self.axes)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b, "add")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b, "subtract")

    def __rsub__(self, other):
        if isinstance(other, AxiomTensor):
            return other.__sub__(self)
        return AxiomTensor(other - self.data, self.axes)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b, "multiply")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b, "divide")

    def __rtruediv__(self, other):
        if isinstance(other, AxiomTensor):
            return other.__truediv__(self)
        return AxiomTensor(other / self.data, self.axes)

    # -------------------------------------------------------------------------
    # Ellipsis
    # -------------------------------------------------------------------------

    def _resolve_ellipsis(self, tokens: Tuple[Any, ...], reference_axes: Tuple[Any, ...]) -> List[Any]:
        if Ellipsis not in tokens:
            return list(tokens)
        if tokens.count(Ellipsis) > 1:
            raise AxiomSyntaxError("Only one Ellipsis (...) is allowed per side.")

        explicit_names = set()
        for t in tokens:
            if t is Ellipsis:
                continue
            if hasattr(t, "source_name"):
                explicit_names.add(t.source_name)
            elif hasattr(t, "axes"):
                explicit_names.update([a.name for a in t.axes])
                explicit_names.add(t.name)
            else:
                explicit_names.add(getattr(t, "name", str(t)))

        hidden_axes = []
        for ax in reference_axes:
            if isinstance(ax, PackedAxis):
                # If the packed name OR any of its child names are on the RHS, do not absorb it
                if ax.name not in explicit_names and not any(a.name in explicit_names for a in ax.axes):
                    hidden_axes.append(ax)
            else:
                if ax.name not in explicit_names:
                    hidden_axes.append(ax)

        resolved = []
        for t in tokens:
            if t is Ellipsis:
                for a in hidden_axes:
                    if isinstance(a, PackedAxis):
                        resolved.append(PackedAxis(*[Axis(x.name, x.size) for x in a.axes]))
                    else:
                        resolved.append(Axis(a.name, a.size))
            else:
                resolved.append(t)
        return resolved

    # -------------------------------------------------------------------------
    # Structural Methods (Phase 3 additions)
    # -------------------------------------------------------------------------

    def _axis_index(self, axis_name: str) -> int:
        for i, a in enumerate(self.axes):
            if a.name == axis_name:
                return i
        raise AxiomShapeError(f"Axis '{axis_name}' not found in tensor.")

    def split(self, axis: Axis, num_chunks: int) -> Tuple["AxiomTensor", ...]:
        """Splits the tensor into distinct physical AxiomTensors along the specified axis."""
        idx = self._axis_index(axis.name)
        chunks = jnp.split(self.data, num_chunks, axis=idx)

        results = []
        for c in chunks:
            new_axes = tuple(
                Axis(a.name, int(c.shape[i]), source_name=getattr(a, "source_name", a.name))
                for i, a in enumerate(self.axes)
            )
            results.append(AxiomTensor(c, new_axes))

        return tuple(results)

    @classmethod
    def concat(cls, tensors: List["AxiomTensor"], axis: Axis) -> "AxiomTensor":
        """Concatenates multiple AxiomTensors along the specified axis."""
        if not tensors:
            raise ValueError("Must provide at least one tensor to concat.")
        if len(tensors) == 1:
            return tensors[0]

        ref_axes = [a.name for a in tensors[0].axes]
        if axis.name not in ref_axes:
            raise AxiomShapeError(f"Concat axis '{axis.name}' not found in reference tensor.")

        idx = ref_axes.index(axis.name)
        data_list = []
        for t in tensors:
            names = [a.name for a in t.axes]
            if names != ref_axes:
                raise AxiomShapeError(f"Cannot concat tensors with mismatched axes: {names} vs {ref_axes}")
            data_list.append(t.data)

        new_data = jnp.concatenate(data_list, axis=idx)

        # Materialize sizes for all axes based on the new concatenated shape
        new_axes = tuple(
            Axis(a.name, int(new_data.shape[i]), source_name=getattr(a, "source_name", a.name))
            for i, a in enumerate(tensors[0].axes)
        )
        return cls(new_data, new_axes)

    def topk(self, axis: Axis, k: int) -> Tuple["AxiomTensor", "AxiomTensor"]:
        """Returns the top K values and indices along a specific axis."""
        idx = self._axis_index(axis.name)
        moved = jnp.moveaxis(self.data, idx, -1)

        values, indices = jax.lax.top_k(moved, k)

        values = jnp.moveaxis(values, -1, idx)
        indices = jnp.moveaxis(indices, -1, idx)

        # Materialize sizes for all axes
        new_axes = tuple(
            Axis(a.name, int(values.shape[i]), source_name=getattr(a, "source_name", a.name))
            for i, a in enumerate(self.axes)
        )

        return AxiomTensor(values, new_axes), AxiomTensor(indices, new_axes)

    # -------------------------------------------------------------------------
    # Param helpers
    # -------------------------------------------------------------------------

    def _get_or_create_param(self, prefix: str, shape, init_fn):
        active_mod = self._require_active_module(prefix)
        param_name = f"{prefix}_{active_mod._axiom_param_counter}"
        object.__setattr__(active_mod, '_axiom_param_counter', active_mod._axiom_param_counter + 1)

        if not getattr(active_mod, "_axiom_initialized", False):
            # Private RNG for params
            seed = id(active_mod) + active_mod._axiom_param_counter
            rng = nnx.Rngs(params=seed).params()
            setattr(active_mod, param_name, nnx.Param(init_fn(rng, shape)))

        param = getattr(active_mod, param_name)
        if hasattr(param, "get_value"):
            return param.get_value()
        return param.value

    def _broadcast_vector(self, vector, idx: int, ndim: int):
        shape = [1] * ndim
        shape[idx] = vector.shape[0]
        return jnp.reshape(vector, shape)

    # -------------------------------------------------------------------------
    # Explicit operand alignment
    # -------------------------------------------------------------------------

    def _align_named_tensor(self, operand: "AxiomTensor", target_axis_names, target_shape):
        operand_names = [a.name for a in operand.axes]
        self._assert_unique_names(operand_names, "explicit operand")

        target_axis_names = list(target_axis_names)
        unknown = [name for name in operand_names if name not in target_axis_names]
        if unknown:
            raise AxiomShapeError(
                f"Explicit operand axes {operand_names} are not a subset of target axes {target_axis_names}."
            )

        ordered_present = [name for name in target_axis_names if name in operand_names]
        perm = [operand_names.index(name) for name in ordered_present]
        data = operand.data if perm == list(range(len(perm))) else jnp.transpose(operand.data, perm)

        size_map = {name: operand.data.shape[operand_names.index(name)] for name in operand_names}
        reshape_shape = [size_map.get(name, 1) for name in target_axis_names]

        for got, want, name in zip(reshape_shape, target_shape, target_axis_names):
            if got not in (1, want):
                raise AxiomShapeError(
                    f"Explicit operand axis '{name}' has size {got}, which is incompatible with target size {want}."
                )

        return jnp.reshape(data, reshape_shape)

    def _align_explicit_operand(self, operand, target_axis_names, target_shape, axis_idx=None, op_name="operand"):
        if isinstance(operand, AxiomTensor):
            return self._align_named_tensor(operand, target_axis_names, target_shape)

        arr = jnp.asarray(operand)
        if arr.ndim == 0:
            return arr
        if arr.shape == tuple(target_shape):
            return arr
        if axis_idx is not None and arr.ndim == 1 and arr.shape[0] == target_shape[axis_idx]:
            return self._broadcast_vector(arr, axis_idx, len(target_shape))

        raise AxiomShapeError(
            f"Explicit {op_name} must be an AxiomTensor, a scalar, an exact-shape array, "
            f"or a 1D array matching axis size {target_shape[axis_idx] if axis_idx is not None else 'N/A'}."
        )

    # -------------------------------------------------------------------------
    # Small math helpers
    # -------------------------------------------------------------------------

    def _default_mask_fill(self, dtype):
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("mask() requires a floating-point tensor.")
        return jnp.array(jnp.finfo(dtype).min, dtype=dtype)

    def _apply_add_bias(
            self,
            current_data,
            current_axis_names,
            axis_idx,
            explicit_bias=None,
            use_implicit=False,
            init_fn=None,
            param_prefix="_axiom_bias",
    ):
        if explicit_bias is not None:
            bias = self._align_explicit_operand(
                explicit_bias,
                current_axis_names,
                current_data.shape,
                axis_idx=axis_idx,
                op_name="bias",
            )
            return current_data + bias.astype(current_data.dtype)

        if use_implicit:
            init = init_fn if init_fn is not None else a_init.zeros
            bias_param = self._get_or_create_param(param_prefix, (current_data.shape[axis_idx],), init)
            return current_data + self._broadcast_vector(bias_param, axis_idx, current_data.ndim).astype(
                current_data.dtype)

        return current_data

    def _apply_mul_gate(
            self,
            current_data,
            current_axis_names,
            axis_idx,
            explicit_gate=None,
            use_implicit=False,
            init_fn=None,
            param_prefix="_axiom_gate",
    ):
        if explicit_gate is not None:
            gate = self._align_explicit_operand(
                explicit_gate,
                current_axis_names,
                current_data.shape,
                axis_idx=axis_idx,
                op_name="gate",
            )
            return current_data * gate.astype(current_data.dtype)

        if use_implicit:
            init = init_fn if init_fn is not None else a_init.ones
            gate_param = self._get_or_create_param(param_prefix, (current_data.shape[axis_idx],), init)
            return current_data * self._broadcast_vector(gate_param, axis_idx, current_data.ndim).astype(
                current_data.dtype)

        return current_data

    def _apply_mul_scale(
            self,
            current_data,
            current_axis_names,
            axis_idx,
            explicit_scale=None,
            use_implicit=False,
            init_fn=None,
            param_prefix="_axiom_scale",
    ):
        if explicit_scale is not None:
            scale = self._align_explicit_operand(
                explicit_scale,
                current_axis_names,
                current_data.shape,
                axis_idx=axis_idx,
                op_name="scale",
            )
            return current_data * scale.astype(current_data.dtype)

        if use_implicit:
            init = init_fn if init_fn is not None else a_init.ones
            scale_param = self._get_or_create_param(param_prefix, (current_data.shape[axis_idx],), init)
            return current_data * self._broadcast_vector(scale_param, axis_idx, current_data.ndim).astype(
                current_data.dtype)

        return current_data

    # -------------------------------------------------------------------------
    # Token helpers
    # -------------------------------------------------------------------------

    def _check_axis_rename_collision(self, current_axis_names, idx: int, new_name: str, where: str):
        old_name = current_axis_names[idx]
        if new_name != old_name and new_name in current_axis_names:
            raise AxiomShapeError(
                f"Duplicate logical axis name '{new_name}' is not allowed ({where}). "
                f"Current axes: {current_axis_names}"
            )

    def _strip_runtime_token(self, token, fallback_size=None):
        if isinstance(token, PackedAxis):
            new_axes = tuple(Axis(a.name, a.size, source_name=a.name) for a in token.axes)
            new = PackedAxis(*new_axes)
            if hasattr(token, "source_name"):
                new.source_name = token.source_name
            return new
        if isinstance(token, Axis):
            return Axis(token.name, token.size if token.size is not None else fallback_size, source_name=token.name)
        raise TypeError(f"Unsupported token type: {type(token)}")

    def _materialize_token_size(self, token, physical_size: int):
        if isinstance(token, Axis):
            return Axis(token.name, physical_size, source_name=token.name)

        if isinstance(token, PackedAxis):
            new_axes = [Axis(a.name, a.size, source_name=a.name) for a in token.axes]

            known_prod = 1
            unknowns = []
            for i, a in enumerate(new_axes):
                if a.size is None:
                    unknowns.append(i)
                else:
                    known_prod *= a.size

            if len(unknowns) == 0:
                total = math.prod(a.size for a in new_axes)
                if total != physical_size:
                    raise AxiomShapeError(
                        f"Packed axis '{token.name}' expects size {total}, but physical size is {physical_size}."
                    )
            elif len(unknowns) == 1:
                if known_prod == 0 or physical_size % known_prod != 0:
                    raise AxiomShapeError(
                        f"Cannot infer packed axis '{token.name}': physical size {physical_size} "
                        f"is not divisible by known factor {known_prod}."
                    )
                i = unknowns[0]
                new_axes[i] = Axis(new_axes[i].name, physical_size // known_prod, source_name=new_axes[i].name)

            new = PackedAxis(*new_axes)
            if hasattr(token, "source_name"):
                new.source_name = token.source_name
            return new

        raise TypeError(f"Unsupported token type in _materialize_token_size: {type(token)}")

    def _finalize_output_axes(self, tokens, current_data):
        return tuple(
            Axis(token.name, int(current_data.shape[i]), source_name=token.name)
            for i, token in enumerate(tokens)
        )

    def _resolve_proj_out_features(self, out_axis, in_features: int):
        if isinstance(out_axis, PackedAxis):
            out_features = 1
            for a in out_axis.axes:
                if a.size is None:
                    raise AxiomShapeError(
                        f"Cannot implicitly infer output size for '{a.name}' during a projection. "
                        "All dimensions in a projection target must have known sizes."
                    )
                out_features *= a.size
            return out_features

        if out_axis.size is None:
            out_axis.size = in_features
        return out_axis.size

    def _conv_dimension_numbers(self, spatial_rank: int):
        if spatial_rank == 1:
            return ("NWC", "WIO", "NWC")
        if spatial_rank == 2:
            return ("NHWC", "HWIO", "NHWC")
        if spatial_rank == 3:
            return ("NDHWC", "DHWIO", "NDHWC")

        alphabet = "QRSTUVWXYZABCDEFGHIJKLMNOP"
        if spatial_rank > len(alphabet):
            raise AxiomShapeError(
                f"Conv with spatial rank {spatial_rank} is too large for symbolic dimension encoding."
            )
        spatial = alphabet[:spatial_rank]
        return ("N" + spatial + "C", spatial + "IO", "N" + spatial + "C")

    def _normalize_conv_int_or_tuple(self, value, rank: int, *, name: str, default: int = 1):
        if value is None:
            return (default,) * rank
        if isinstance(value, int):
            if value <= 0:
                raise AxiomShapeError(f"conv(...): {name} must be positive.")
            return (value,) * rank
        if isinstance(value, (tuple, list)):
            if len(value) != rank:
                raise AxiomShapeError(
                    f"conv(...): {name} rank mismatch: expected {rank}, got {len(value)}."
                )
            if not all(isinstance(v, int) and v > 0 for v in value):
                raise AxiomShapeError(f"conv(...): all {name} values must be positive integers.")
            return tuple(value)
        raise AxiomShapeError(
            f"conv(...): {name} must be an int or a tuple/list of length {rank}."
        )

    def _normalize_explicit_conv_pad(self, pad, rank: int):
        # 1D shorthand: (left, right)
        if (
                rank == 1
                and isinstance(pad, (tuple, list))
                and len(pad) == 2
                and all(isinstance(x, int) for x in pad)
        ):
            return (tuple(pad),)

        if not isinstance(pad, (tuple, list)):
            raise AxiomShapeError(
                "conv(...): explicit pad must be a string-free tuple/list of (low, high) pairs."
            )

        pad = tuple(tuple(p) for p in pad)
        if len(pad) != rank or any(len(p) != 2 for p in pad):
            raise AxiomShapeError(
                f"conv(...): explicit pad must have one (low, high) pair per spatial dim. "
                f"Expected rank {rank}, got {pad}."
            )
        if not all(isinstance(x, int) and x >= 0 for pair in pad for x in pair):
            raise AxiomShapeError("conv(...): explicit pad values must be non-negative integers.")
        return pad

    def _resolve_conv_kernel(self, kernel, size_map: dict):
        if isinstance(kernel, Axis):
            kernel = self._resolve_token_sizes(kernel, size_map)
            if kernel.ops:
                raise AxiomShapeError("conv(kernel=...): kernel axes may not carry ops.")
            if kernel.size is None:
                raise AxiomShapeError("conv(kernel=...): named kernel axes must have known sizes.")
            return (int(kernel.size),), [Axis(kernel.name, int(kernel.size), source_name=kernel.source_name)]

        if isinstance(kernel, PackedAxis):
            kernel = self._resolve_token_sizes(kernel, size_map)
            axes = []
            sizes = []
            for a in kernel.axes:
                if a.ops:
                    raise AxiomShapeError("conv(kernel=...): kernel axes may not carry ops.")
                if a.size is None:
                    raise AxiomShapeError("conv(kernel=...): named kernel axes must have known sizes.")
                axes.append(Axis(a.name, int(a.size), source_name=a.source_name))
                sizes.append(int(a.size))
            return tuple(sizes), axes

        if isinstance(kernel, int):
            if kernel <= 0:
                raise AxiomShapeError("conv(kernel=...): integer kernel size must be positive.")
            return (kernel,), None

        if isinstance(kernel, (tuple, list)):
            if not kernel:
                raise AxiomShapeError("conv(kernel=...): kernel tuple must be non-empty.")
            if not all(isinstance(k, int) and k > 0 for k in kernel):
                raise AxiomShapeError("conv(kernel=...): all kernel sizes must be positive integers.")
            return tuple(kernel), None

        raise AxiomShapeError(
            "conv(kernel=...): kernel must be an int, a tuple of ints, an Axis, or a PackedAxis."
        )

    def _strip_conv_domain_axis(self, axis: Axis):
        mode = None
        stride = 1
        dilation = 1

        for op in axis.ops:
            if isinstance(op, ConvModeOp):
                if mode is not None:
                    raise AxiomShapeError(
                        f"conv(over=...): axis '{axis.name}' has multiple conv modes attached."
                    )
                mode = op.mode
            elif isinstance(op, ConvStrideOp):
                if stride != 1:
                    raise AxiomShapeError(
                        f"conv(over=...): axis '{axis.name}' has multiple stride(...) modifiers."
                    )
                stride = op.value
            elif isinstance(op, ConvDilationOp):
                if dilation != 1:
                    raise AxiomShapeError(
                        f"conv(over=...): axis '{axis.name}' has multiple dilate(...) modifiers."
                    )
                dilation = op.value
            else:
                raise AxiomShapeError(
                    f"conv(over=...): axis '{axis.name}' may only carry conv-domain modifiers "
                    f"(same()/valid()/causal()/stride()/dilate())."
                )

        return Axis(axis.name, axis.size, source_name=axis.source_name), mode, stride, dilation

    def _build_conv_padding(self, modes, kernel_sizes, dilations):
        padding = []
        for mode, k, d in zip(modes, kernel_sizes, dilations):
            mode = "same" if mode is None else mode
            total = d * (k - 1)

            if mode == "same":
                low = total // 2
                high = total - low
            elif mode == "valid":
                low = 0
                high = 0
            elif mode == "causal":
                low = total
                high = 0
            else:
                raise AxiomShapeError(f"Unknown conv mode '{mode}'.")

            padding.append((low, high))
        return tuple(padding)

    def _resolve_conv_over(self, op: ConvOp, current_axis_names, feature_idx: int, kernel_sizes, size_map: dict):
        spatial_rank = len(kernel_sizes)

        # Legacy mode: infer the last `spatial_rank` axes immediately to the left of the feature axis.
        if op.over is None:
            if feature_idx < spatial_rank:
                raise AxiomShapeError(
                    f"Legacy conv inference failed: feature axis index {feature_idx} does not have "
                    f"{spatial_rank} spatial axes immediately to its left."
                )

            domain_indices = tuple(range(feature_idx - spatial_rank, feature_idx))
            domain_axes = [
                Axis(current_axis_names[i], int(size_map[current_axis_names[i]]), source_name=current_axis_names[i])
                for i in domain_indices
            ]

            strides = self._normalize_conv_int_or_tuple(
                op.legacy_strides, spatial_rank, name="legacy strides", default=1
            )
            dilations = self._normalize_conv_int_or_tuple(
                op.legacy_dilation, spatial_rank, name="legacy dilation", default=1
            )

            if op.legacy_padding is None:
                padding = self._build_conv_padding([None] * spatial_rank, kernel_sizes, dilations)
            elif isinstance(op.legacy_padding, str):
                p = op.legacy_padding.lower()
                if p == "same":
                    padding = self._build_conv_padding(["same"] * spatial_rank, kernel_sizes, dilations)
                elif p == "valid":
                    padding = self._build_conv_padding(["valid"] * spatial_rank, kernel_sizes, dilations)
                elif p == "causal":
                    if spatial_rank != 1:
                        raise AxiomShapeError("Legacy padding='causal' is only valid for 1D convolution.")
                    padding = self._build_conv_padding(["causal"], kernel_sizes, dilations)
                else:
                    raise AxiomShapeError(
                        f"Unsupported legacy conv padding string '{op.legacy_padding}'. "
                        f"Use 'same', 'valid', or 'causal'."
                    )
            else:
                padding = self._normalize_explicit_conv_pad(op.legacy_padding, spatial_rank)

            return domain_axes, domain_indices, strides, dilations, padding

        # Canonical mode: over=Axis | PackedAxis with domain metadata on the axes themselves.
        resolved_over = self._resolve_token_sizes(op.over, size_map)
        flat_over = list(resolved_over.axes) if isinstance(resolved_over, PackedAxis) else [resolved_over]

        if len(flat_over) != spatial_rank:
            raise AxiomShapeError(
                f"conv(...): kernel rank {spatial_rank} does not match the number of domain axes in over={len(flat_over)}."
            )

        domain_axes = []
        domain_indices = []
        modes = []
        strides = []
        dilations = []

        for a in flat_over:
            base_axis, mode, stride, dilation = self._strip_conv_domain_axis(a)

            if base_axis.name not in current_axis_names:
                raise AxiomShapeError(
                    f"conv(...): over-axis '{base_axis.name}' is not present in the current tensor axes {current_axis_names}."
                )

            axis_idx = current_axis_names.index(base_axis.name)
            domain_axes.append(base_axis)
            domain_indices.append(axis_idx)
            modes.append(mode)
            strides.append(stride)
            dilations.append(dilation)

        if len(set(domain_indices)) != len(domain_indices):
            raise AxiomShapeError("conv(...): over=... may not reference the same axis more than once.")
        if feature_idx in domain_indices:
            raise AxiomShapeError("conv(...): the feature axis may not also appear in over=...")

        if op.pad is not None:
            if any(mode is not None for mode in modes):
                raise AxiomShapeError(
                    "conv(...): do not combine explicit pad= with over-axis modes like same()/valid()/causal()."
                )
            padding = self._normalize_explicit_conv_pad(op.pad, spatial_rank)
        else:
            padding = self._build_conv_padding(modes, kernel_sizes, dilations)

        return domain_axes, tuple(domain_indices), tuple(strides), tuple(dilations), padding

    def _validate_named_conv_weight(
            self,
            weight: "AxiomTensor",
            kernel_axes,
            in_axis_name: str,
            in_features: int,
            out_axis_name: str,
            out_features: int,
            groups,
    ):
        if kernel_axes is None:
            return
        if groups != 1:
            # Grouped explicit kernels are still shape-validated by the backend path.
            return
        if not isinstance(weight, AxiomTensor):
            return

        expected_names = [a.name for a in kernel_axes] + [in_axis_name, out_axis_name]
        actual_names = [a.name for a in weight.axes]

        if actual_names != expected_names:
            raise AxiomShapeError(
                f"Explicit conv weight axes mismatch: expected {expected_names}, got {actual_names}."
            )

        expected_sizes = [int(a.size) for a in kernel_axes] + [int(in_features), int(out_features)]
        actual_sizes = list(weight.data.shape)

        for name, expected, got in zip(expected_names, expected_sizes, actual_sizes):
            if expected != got:
                raise AxiomShapeError(
                    f"Explicit conv weight axis '{name}' size mismatch: expected {expected}, got {got}."
                )

    def _run_lax_conv(
            self,
            current_data,
            current_axis_names,
            feature_idx: int,
            domain_indices,
            kernel,
            kernel_sizes,
            strides,
            dilations,
            padding,
            groups,
    ):
        spatial_rank = len(kernel_sizes)
        if len(domain_indices) != spatial_rank:
            raise AxiomShapeError(
                f"Internal conv error: got {len(domain_indices)} domain axes for kernel rank {spatial_rank}."
            )

        batch_indices = [i for i in range(current_data.ndim) if i not in domain_indices and i != feature_idx]
        perm = batch_indices + list(domain_indices) + [feature_idx]

        x = current_data if perm == list(range(current_data.ndim)) else jnp.transpose(current_data, perm)

        batch_shape = tuple(x.shape[:len(batch_indices)])
        spatial_shape = tuple(x.shape[len(batch_indices):-1])

        if len(spatial_shape) != spatial_rank:
            raise AxiomShapeError(
                f"Internal conv error: expected {spatial_rank} spatial dims, got {len(spatial_shape)}."
            )

        if len(batch_shape) == 0:
            x = x.reshape((1,) + spatial_shape + (x.shape[-1],))
        elif len(batch_shape) > 1:
            x = x.reshape((math.prod(batch_shape),) + spatial_shape + (x.shape[-1],))

        in_features = int(x.shape[-1])
        groups_resolved = in_features if groups in (-1, "depthwise") else int(groups)

        if in_features % groups_resolved != 0:
            raise AxiomShapeError(
                f"Conv groups mismatch: input features {in_features} must be divisible by groups={groups_resolved}."
            )

        kernel = jnp.asarray(kernel)
        if kernel.ndim != spatial_rank + 2:
            raise AxiomShapeError(
                f"Conv kernel must have rank {spatial_rank + 2}, got {kernel.ndim}."
            )

        if tuple(kernel.shape[:spatial_rank]) != tuple(kernel_sizes):
            raise AxiomShapeError(
                f"Conv kernel shape mismatch: expected spatial kernel {kernel_sizes}, "
                f"got {kernel.shape[:spatial_rank]}."
            )

        expected_in_per_group = in_features // groups_resolved
        if int(kernel.shape[-2]) != expected_in_per_group:
            raise AxiomShapeError(
                f"Conv kernel in_features mismatch: expected {expected_in_per_group} "
                f"(input features per group), got {kernel.shape[-2]}."
            )

        out_features = int(kernel.shape[-1])
        if out_features % groups_resolved != 0:
            raise AxiomShapeError(
                f"Conv output features {out_features} must be divisible by groups={groups_resolved}."
            )

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=strides,
            padding=padding,
            rhs_dilation=dilations,
            dimension_numbers=self._conv_dimension_numbers(spatial_rank),
            feature_group_count=groups_resolved,
        )

        if len(batch_shape) == 0:
            y = y.reshape(y.shape[1:])
        elif len(batch_shape) > 1:
            y = y.reshape(batch_shape + y.shape[1:])

        inv_perm = [0] * len(perm)
        for canon_idx, orig_idx in enumerate(perm):
            inv_perm[orig_idx] = canon_idx

        if inv_perm != list(range(len(inv_perm))):
            y = jnp.transpose(y, inv_perm)

        return y, out_features

    def _apply_explicit_conv(
            self,
            current_data,
            current_axis_names,
            feature_idx: int,
            domain_indices,
            kernel_sizes,
            kernel_axes,
            strides,
            dilations,
            padding,
            op: ConvOp,
            out_axis_name: str,
    ):
        if isinstance(op.weight, AxiomTensor):
            inferred_out = int(op.weight.data.shape[-1])
            self._validate_named_conv_weight(
                op.weight,
                kernel_axes,
                current_axis_names[feature_idx],
                int(current_data.shape[feature_idx]),
                out_axis_name,
                inferred_out,
                op.groups,
            )
            kernel = op.weight.data
        else:
            kernel = op.weight

        return self._run_lax_conv(
            current_data,
            current_axis_names,
            feature_idx,
            domain_indices,
            kernel,
            kernel_sizes,
            strides,
            dilations,
            padding,
            op.groups,
        )

    def _apply_implicit_conv(
            self,
            current_data,
            current_axis_names,
            feature_idx: int,
            domain_indices,
            kernel_sizes,
            strides,
            dilations,
            padding,
            op: ConvOp,
            out_features: int,
    ):
        in_features = int(current_data.shape[feature_idx])
        groups_resolved = in_features if op.groups in (-1, "depthwise") else int(op.groups)

        if in_features % groups_resolved != 0:
            raise AxiomShapeError(
                f"Implicit conv groups mismatch: input features {in_features} "
                f"must be divisible by groups={groups_resolved}."
            )
        if out_features % groups_resolved != 0:
            raise AxiomShapeError(
                f"Implicit conv output features {out_features} "
                f"must be divisible by groups={groups_resolved}."
            )

        kernel_shape = tuple(kernel_sizes) + (in_features // groups_resolved, out_features)
        k_init = op.kernel_init if op.kernel_init is not None else a_init.default_kernel_init
        kernel = self._get_or_create_param("_axiom_conv_kernel", kernel_shape, k_init)

        return self._run_lax_conv(
            current_data,
            current_axis_names,
            feature_idx,
            domain_indices,
            kernel,
            kernel_sizes,
            strides,
            dilations,
            padding,
            op.groups,
        )

    def _expand_packed_token_for_rhs(self, token: PackedAxis, physical_size: int):
        known_prod = 1
        unknowns = []
        for a in token.axes:
            if a.size is None:
                unknowns.append(a)
            else:
                known_prod *= a.size

        if len(unknowns) > 1:
            return None

        if len(unknowns) == 1:
            if physical_size % known_prod != 0:
                raise AxiomShapeError(
                    f"Cannot unpack packed axis '{token.name}': physical size {physical_size} is not divisible by known factor {known_prod}."
                )
            unknowns[0].size = physical_size // known_prod
        else:
            total = math.prod(a.size for a in token.axes)
            if total != physical_size:
                raise AxiomShapeError(
                    f"Packed axis '{token.name}' expects size {total}, but physical size is {physical_size}."
                )

        return [Axis(a.name, a.size, source_name=a.name) for a in token.axes]

    def _normalize_surviving_for_rhs(self, current_data, surviving_tokens):
        new_shape = []
        new_tokens = []

        for i, token in enumerate(surviving_tokens):
            physical = int(current_data.shape[i])

            if isinstance(token, PackedAxis):
                expanded = self._expand_packed_token_for_rhs(token, physical)
                if expanded is None:
                    new_shape.append(physical)
                    new = PackedAxis(*[Axis(a.name, a.size, source_name=a.name) for a in token.axes])
                    if hasattr(token, "source_name"):
                        new.source_name = token.source_name
                    new_tokens.append(new)
                else:
                    for child in expanded:
                        new_shape.append(child.size)
                        new_tokens.append(child)
            else:
                new_tokens.append(Axis(token.name, physical, source_name=token.name))
                new_shape.append(physical)

        current_data = jnp.reshape(current_data, new_shape)
        self._assert_unique_names([t.name for t in new_tokens], "RHS-prep normalized surviving axes")
        return current_data, new_tokens

    # -------------------------------------------------------------------------
    # Op execution
    # -------------------------------------------------------------------------

    def _execute_ops(self, current_data, current_axis_names, token, idx):
        current_token = self._strip_runtime_token(token, fallback_size=int(current_data.shape[idx]))

        for op in token.ops:
            if isinstance(op, (ConvModeOp, ConvStrideOp, ConvDilationOp)):
                raise AxiomSyntaxError(
                    "same()/valid()/causal()/stride()/dilate() are conv-domain modifiers "
                    "and may only be used inside conv(over=...)."
                )

            if op == "relu":
                current_data = jax.nn.relu(current_data)

        for op in token.ops:
            if op == "relu":
                current_data = jax.nn.relu(current_data)

            elif op == "silu":
                current_data = jax.nn.silu(current_data)

            elif op == "gelu":
                current_data = jax.nn.gelu(current_data)

            elif op == "sigmoid":
                current_data = jax.nn.sigmoid(current_data)

            elif op == "tanh":
                current_data = jnp.tanh(current_data)

            elif op == "softmax":
                current_data = jax.nn.softmax(current_data, axis=idx)

            elif op == "softplus":
                current_data = jax.nn.softplus(current_data)

            # --- Phase 1: Pointwise ---
            elif op == "exp":
                current_data = jnp.exp(current_data)
            elif op == "log":
                current_data = jnp.log(current_data)
            elif op == "abs":
                current_data = jnp.abs(current_data)
            elif op == "rsqrt":
                current_data = jax.lax.rsqrt(current_data)
            elif op == "swiglu":
                # Splitting automatically halves the axis size, the _materialize_token_size
                # function at the end of the loop will catch the new size perfectly.
                chunk1, chunk2 = jnp.split(current_data, 2, axis=idx)
                current_data = jax.nn.silu(chunk1) * chunk2

            elif op == "square":
                current_data = jnp.square(current_data)
            elif isinstance(op, PowOp):
                current_data = jnp.power(current_data, op.exponent)
            elif op == "round":
                current_data = jnp.round(current_data)
            elif op == "floor":
                current_data = jnp.floor(current_data)
            elif op == "ceil":
                current_data = jnp.ceil(current_data)
            elif op == "sin":
                current_data = jnp.sin(current_data)
            elif op == "cos":
                current_data = jnp.cos(current_data)


            elif isinstance(op, ClampOp):
                # Updated to use JAX's modern 'min' and 'max' kwargs
                current_data = jnp.clip(current_data, min=op.min_val, max=op.max_val)


            elif isinstance(op, AssocScanOp):
                scan_elems = [current_data]

                # Align any extra input tensors to the current sequence axis
                for extra in op.inputs:
                    extra_names = [a.name for a in extra.axes]
                    e_idx = extra_names.index(current_token.name)
                    aligned_extra = jnp.moveaxis(extra.data, e_idx, idx)
                    scan_elems.append(aligned_extra)

                # JAX handles PyTrees natively! It will return a tuple of (x_out, a_out)
                scanned_out = jax.lax.associative_scan(
                    op.fn,
                    tuple(scan_elems),
                    reverse=op.reverse,
                    axis=idx
                )

                # The active routing chain continues with the primary tensor (x_out)
                current_data = scanned_out[0]

            # --- Phase 3: Control & Scatter ---
            elif isinstance(op, StopGradientOp):
                current_data = jax.lax.stop_gradient(current_data)

            elif isinstance(op, ScatterOp):
                indices_data = op.indices.data if hasattr(op.indices, "data") else op.indices
                updates_data = op.updates.data if hasattr(op.updates, "data") else op.updates

                # Build n-dimensional slice tuple to target the specific axis
                slices = [slice(None)] * current_data.ndim
                slices[idx] = indices_data

                if op.mode == "add":
                    current_data = current_data.at[tuple(slices)].add(updates_data)
                else:
                    current_data = current_data.at[tuple(slices)].set(updates_data)

            elif isinstance(op, WhereOp):
                cond_data = self._align_explicit_operand(
                    op.condition, current_axis_names, current_data.shape, axis_idx=None, op_name="where condition"
                )
                false_data = self._align_explicit_operand(
                    op.false_tensor, current_axis_names, current_data.shape, axis_idx=None, op_name="where false_tensor"
                )
                current_data = jnp.where(cond_data, current_data, false_data)

            elif isinstance(op, FillOp):
                # Modernized to idiomatic JAX
                current_data = jnp.full_like(current_data, op.value)

            elif isinstance(op, RollOp):
                shift = op.shift.data if hasattr(op.shift, "data") else op.shift
                current_data = jnp.roll(current_data, shift, axis=idx)

            elif isinstance(op, PadOp):
                pad_config = [(0, 0)] * current_data.ndim
                pad_config[idx] = op.pad_width
                current_data = jnp.pad(current_data, pad_config, mode=op.mode, constant_values=op.value)
                # Materialize the new padded size
                current_token = Axis(current_token.name, int(current_data.shape[idx]),
                                     source_name=current_token.source_name)

            elif isinstance(op, GatherOp):
                indices_data = op.indices.data

                # 1. Perform standard JAX take
                current_data = jnp.take(current_data, indices_data, axis=idx)

                # 2. Rank has increased! Flatten the newly inserted dimensions temporarily.
                # This ensures the LHS router loop still treats this operation as returning a single token.
                flattened_dim_size = math.prod(indices_data.shape)
                new_shape = list(current_data.shape)
                new_shape = new_shape[:idx] + [flattened_dim_size] + new_shape[idx + indices_data.ndim:]
                current_data = jnp.reshape(current_data, new_shape)

                # 3. Construct a PackedAxis representing the newly gathered dimensions
                new_axes = op.indices.axes
                packed_name = "&".join(a.name for a in new_axes)
                current_axis_names[idx] = packed_name
                current_token = PackedAxis(*new_axes)

            elif isinstance(op, CastOp):
                current_data = current_data.astype(op.dtype)

            elif isinstance(op, MaskOp):
                if idx == 0:
                    raise AxiomShapeError("mask() with other_axis='left' requires an axis directly to the left.")
                other_idx = idx - 1
                row_len = current_data.shape[other_idx]
                col_len = current_data.shape[idx]

                row_shape = [1] * current_data.ndim
                row_shape[other_idx] = row_len
                col_shape = [1] * current_data.ndim
                col_shape[idx] = col_len

                row_idx = jnp.arange(row_len).reshape(row_shape)
                col_idx = jnp.arange(col_len).reshape(col_shape)

                if op.kind == "tril":
                    keep = row_idx >= col_idx
                else:
                    keep = row_idx <= col_idx

                fill_value = (
                    self._default_mask_fill(current_data.dtype)
                    if op.fill_value is None
                    else jnp.asarray(op.fill_value, dtype=current_data.dtype)
                )
                current_data = jnp.where(keep, current_data, fill_value)

            elif isinstance(op, GateOp):
                current_data = self._apply_mul_gate(
                    current_data,
                    current_axis_names,
                    idx,
                    explicit_gate=op.tensor,
                    use_implicit=(op.tensor is None),
                    init_fn=op.init_fn,
                    param_prefix="_axiom_gate",
                )

            elif isinstance(op, BiasOp):
                current_data = self._apply_add_bias(
                    current_data,
                    current_axis_names,
                    idx,
                    explicit_bias=op.tensor,
                    use_implicit=(op.tensor is None),
                    init_fn=op.init_fn,
                    param_prefix="_axiom_bias",
                )

            elif isinstance(op, NormOp):
                compute_dtype = jnp.promote_types(current_data.dtype, jnp.float32)
                moved = jnp.moveaxis(current_data, idx, -1).astype(compute_dtype)

                if op.norm_type == "rms":
                    rms = jnp.sqrt(jnp.mean(jnp.square(moved), axis=-1, keepdims=True) + op.eps)
                    normed = moved / rms
                else:
                    mean = jnp.mean(moved, axis=-1, keepdims=True)
                    var = jnp.var(moved, axis=-1, keepdims=True)
                    normed = (moved - mean) / jnp.sqrt(var + op.eps)

                current_data = jnp.moveaxis(normed, -1, idx).astype(current_data.dtype)

                current_data = self._apply_mul_scale(
                    current_data,
                    current_axis_names,
                    idx,
                    explicit_scale=op.scale,
                    use_implicit=(op.scale is None and op.use_scale),
                    init_fn=op.init_scale,
                    param_prefix="_axiom_norm_scale",
                )

                if op.norm_type == "layer":
                    current_data = self._apply_add_bias(
                        current_data,
                        current_axis_names,
                        idx,
                        explicit_bias=op.bias,
                        use_implicit=(op.bias is None and op.use_bias),
                        init_fn=op.init_bias,
                        param_prefix="_axiom_norm_bias",
                    )

            elif isinstance(op, DropoutOp):
                active_mod = self._require_active_module("Dropout")
                param_name = f"_axiom_drop_{active_mod._axiom_param_counter}"
                object.__setattr__(active_mod, '_axiom_param_counter', active_mod._axiom_param_counter + 1)

                if not getattr(active_mod, "_axiom_initialized", False):
                    # Native assignment of our custom remat-safe class
                    setattr(active_mod, param_name, AxiomDropout(rate=op.rate))

                drop_layer = getattr(active_mod, param_name)
                current_data = drop_layer(current_data)

            elif isinstance(op, ScanOp):
                scan_main = jnp.swapaxes(current_data, 0, idx)
                scan_extras = []

                for extra in op.inputs:
                    extra_names = [a.name for a in extra.axes]
                    e_idx = extra_names.index(current_token.name)
                    scan_extras.append(jnp.swapaxes(extra.data, 0, e_idx))

                # Handle the explicit initialization state
                if op.init is not None:
                    init_state = op.init.data if hasattr(op.init, "data") else op.init
                else:
                    init_state = jnp.zeros_like(scan_main[0])

                def scan_body(carry, xs):
                    new_state, out_y = op.fn(carry, (xs[0], *xs[1:]))
                    new_state = new_state.astype(carry.dtype)
                    return new_state, out_y

                _, scanned_out = jax.lax.scan(scan_body, init_state, (scan_main, *scan_extras))
                current_data = jnp.swapaxes(scanned_out, 0, idx)

            elif isinstance(op, AttendOp):
                q_seq_name = current_axis_names[idx]
                dim_name = op.dim.name if hasattr(op.dim, "name") else str(op.dim)
                if dim_name not in current_axis_names:
                    raise AxiomShapeError(f".attend() requires feature dimension '{dim_name}' in the query tensor.")

                # 1. Identify query layout
                batch_names = [n for n in current_axis_names if n not in (q_seq_name, dim_name)]
                q_target_layout = batch_names + [q_seq_name, dim_name]
                q_perm = [current_axis_names.index(n) for n in q_target_layout]
                q_aligned = jnp.transpose(current_data, q_perm)

                # Flatten batch & head names into a single dimension, and insert num_heads=1
                # Shape becomes: (batch_flat, seq_len, 1, dim_size)
                batch_size_flat = math.prod(q_aligned.shape[:-2]) if batch_names else 1
                q_seq_len = q_aligned.shape[-2]
                dim_size = q_aligned.shape[-1]
                q_reshaped = jnp.reshape(q_aligned, (batch_size_flat, q_seq_len, 1, dim_size))

                # 2. Identify keys/values layout
                if not isinstance(op.keys, AxiomTensor) or not isinstance(op.values, AxiomTensor):
                    raise TypeError(".attend() requires explicit keys and values to be AxiomTensors.")

                k_names = [a.name for a in op.keys.axes]
                v_names = [a.name for a in op.values.axes]
                kv_seq_candidates = [n for n in k_names if n not in batch_names and n != dim_name]

                if len(kv_seq_candidates) != 1:
                    raise AxiomShapeError(
                        f"Could not automatically infer the kv-sequence axis. "
                        f"Keys: {k_names}, Batch: {batch_names}, Feature: {dim_name}"
                    )

                kv_seq_name = kv_seq_candidates[0]
                kv_target_layout = batch_names + [kv_seq_name, dim_name]
                if set(k_names) != set(kv_target_layout):
                    raise AxiomShapeError(f"Keys tensor must have exactly axes {kv_target_layout}. Got {k_names}.")

                if set(v_names) != set(kv_target_layout):
                    raise AxiomShapeError(f"Values tensor must have exactly axes {kv_target_layout}. Got {v_names}.")

                k_aligned = jnp.transpose(op.keys.data, [k_names.index(n) for n in kv_target_layout])
                kv_seq_len = k_aligned.shape[-2]
                k_reshaped = jnp.reshape(k_aligned, (batch_size_flat, kv_seq_len, 1, dim_size))
                v_aligned = jnp.transpose(op.values.data, [v_names.index(n) for n in kv_target_layout])
                v_reshaped = jnp.reshape(v_aligned, (batch_size_flat, kv_seq_len, 1, dim_size))

                # 3. Create boolean causal mask if requested
                mask = None

                if op.is_causal:
                    mask = jnp.arange(q_seq_len)[:, None] >= jnp.arange(kv_seq_len)[None, :]

                # 4. Execute FlashAttention Backend
                # JAX dot_product_attention now gets the exact 4D shape it strictly requires
                attended = jax.nn.dot_product_attention(q_reshaped, k_reshaped, v_reshaped, mask=mask)

                # 5. Restore user's original logical layout
                attended_unflattened = jnp.reshape(attended, q_aligned.shape)
                restore_perm = [q_target_layout.index(n) for n in current_axis_names]
                current_data = jnp.transpose(attended_unflattened, restore_perm)

            elif isinstance(op, ProjOp):
                # Dynamically resolve any symbolic output sizes right before executing
                size_map = {name: int(current_data.shape[i]) for i, name in enumerate(current_axis_names)}
                resolved_out_axis = self._resolve_token_sizes(op.out_axis, size_map)

                in_features = int(current_data.shape[idx])

                self._check_axis_rename_collision(
                    current_axis_names,
                    idx,
                    resolved_out_axis.name,
                    "projection output",
                )

                if op.weight is not None and isinstance(resolved_out_axis, PackedAxis):
                    raise AxiomShapeError(
                        "Explicit proj(weight=...) does not currently support out=PackedAxis. "
                        "Project to a flat Axis, then unpack or route afterward."
                    )

                out_features = self._resolve_proj_out_features(resolved_out_axis, in_features)

                if op.weight is None:
                    active_mod = self._require_active_module("Implicit .proj()")
                    param_name = f"_axiom_proj_{active_mod._axiom_param_counter}"
                    object.__setattr__(active_mod, '_axiom_param_counter', active_mod._axiom_param_counter + 1)

                    if not getattr(active_mod, "_axiom_initialized", False):
                        seed = id(active_mod) + active_mod._axiom_param_counter
                        k_init = op.kernel_init if op.kernel_init is not None else a_init.default_kernel_init
                        p_dtype = op.dtype if op.dtype is not None else jnp.float32
                        setattr(
                            active_mod,
                            param_name,
                            nnx.Linear(
                                in_features,
                                out_features,
                                use_bias=False,
                                kernel_init=k_init,
                                dtype=p_dtype,
                                param_dtype=p_dtype,
                                rngs=nnx.Rngs(params=seed),  # Private RNG
                            ),
                        )

                    linear_layer = getattr(active_mod, param_name)
                    current_data = jnp.moveaxis(current_data, idx, -1)
                    current_data = linear_layer(current_data)
                    current_data = jnp.moveaxis(current_data, -1, idx)

                else:
                    if not isinstance(op.weight, AxiomTensor):
                        raise TypeError("Explicit projection weight must be an AxiomTensor.")

                    lhs_names = current_axis_names.copy()
                    rhs_names = [a.name for a in op.weight.axes]
                    self._assert_unique_names(rhs_names, "explicit projection weight")

                    contracted_name = current_axis_names[idx]
                    out_name = resolved_out_axis.name
                    out_names = [out_name if n == contracted_name else n for n in lhs_names]

                    unique_names = self._ordered_unique(lhs_names + rhs_names + out_names)
                    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    if len(unique_names) > len(chars):
                        raise AxiomShapeError("Too many unique axis names for einsum projection.")

                    n2c = {name: chars[i] for i, name in enumerate(unique_names)}
                    lhs_str = "".join(n2c[n] for n in lhs_names)
                    rhs_str = "".join(n2c[n] for n in rhs_names)
                    out_str = "".join(n2c[n] for n in out_names)

                    current_data = jnp.einsum(f"{lhs_str},{rhs_str}->{out_str}", current_data, op.weight.data)
                    current_axis_names = out_names
                    idx = current_axis_names.index(out_name)
                    out_features = int(current_data.shape[idx])

                current_axis_names[idx] = resolved_out_axis.name
                self._assert_unique_names(current_axis_names, "projection output")

                if isinstance(resolved_out_axis, PackedAxis):
                    current_token = PackedAxis(
                        *[Axis(a.name, a.size, source_name=a.name) for a in resolved_out_axis.axes])
                    if hasattr(resolved_out_axis, "source_name"):
                        current_token.source_name = resolved_out_axis.source_name
                else:
                    current_token = Axis(resolved_out_axis.name, out_features, source_name=resolved_out_axis.name)

                current_data = self._apply_add_bias(
                    current_data,
                    current_axis_names,
                    idx,
                    explicit_bias=op.bias,
                    use_implicit=(op.bias is None and op.use_bias),
                    init_fn=op.bias_init,
                    param_prefix="_axiom_proj_bias",
                )



            elif isinstance(op, ConvOp):
                size_map = {name: int(current_data.shape[i]) for i, name in enumerate(current_axis_names)}
                resolved_out_axis = self._resolve_token_sizes(op.out_axis, size_map)

                if isinstance(resolved_out_axis, PackedAxis):
                    raise AxiomShapeError("conv(...): out must be a single Axis, not a PackedAxis.")

                in_features = int(current_data.shape[idx])
                if resolved_out_axis.size is None:
                    resolved_out_axis = Axis(
                        resolved_out_axis.name,
                        in_features,
                        source_name=resolved_out_axis.source_name,
                    )

                self._check_axis_rename_collision(
                    current_axis_names,
                    idx,
                    resolved_out_axis.name,
                    "conv output",
                )

                kernel_sizes, kernel_axes = self._resolve_conv_kernel(op.kernel, size_map)
                domain_axes, domain_indices, strides, dilations, padding = self._resolve_conv_over(
                    op,
                    current_axis_names,
                    idx,
                    kernel_sizes,
                    size_map,
                )

                if op.weight is None:
                    out_features = int(resolved_out_axis.size)
                    current_data, actual_out = self._apply_implicit_conv(
                        current_data,
                        current_axis_names,
                        idx,
                        domain_indices,
                        kernel_sizes,
                        strides,
                        dilations,
                        padding,
                        op,
                        out_features,
                    )

                    if actual_out != out_features:
                        raise AxiomShapeError(
                            f"Implicit conv internal mismatch: expected {out_features} output features, got {actual_out}."
                        )
                else:
                    current_data, out_features = self._apply_explicit_conv(
                        current_data,
                        current_axis_names,
                        idx,
                        domain_indices,
                        kernel_sizes,
                        kernel_axes,
                        strides,
                        dilations,
                        padding,
                        op,
                        resolved_out_axis.name,
                    )

                    if resolved_out_axis.size is not None and int(resolved_out_axis.size) != out_features:
                        raise AxiomShapeError(
                            f"Explicit conv output feature mismatch: expected {resolved_out_axis.size}, got {out_features}."
                        )

                current_axis_names[idx] = resolved_out_axis.name
                self._assert_unique_names(current_axis_names, "conv output")
                current_token = Axis(
                    resolved_out_axis.name,
                    out_features,
                    source_name=resolved_out_axis.name,
                )

                current_data = self._apply_add_bias(
                    current_data,
                    current_axis_names,
                    idx,
                    explicit_bias=op.bias,
                    use_implicit=(op.bias is None and op.use_bias),
                    init_fn=op.bias_init,
                    param_prefix="_axiom_conv_bias",
                )

        return current_data, current_axis_names, current_token

    # -------------------------------------------------------------------------
    # Main routing
    # -------------------------------------------------------------------------

    def __getitem__(self, tokens: Any) -> "AxiomTensor":
        if not isinstance(tokens, tuple):
            tokens = (tokens,)

        arrow_count = tokens.count("->")
        if arrow_count > 1:
            raise AxiomSyntaxError("Multiple '->' operators found.")

        if arrow_count == 1:
            split_idx = tokens.index("->")
            lhs_tokens, rhs_tokens = tokens[:split_idx], tokens[split_idx + 1:]
        else:
            lhs_tokens, rhs_tokens = tokens, None

        lhs_resolved = self._resolve_ellipsis(lhs_tokens, self.axes)

        # Intercept and dynamically resolve any SymbolicSize objects parsed from the LHS
        size_map = {a.name: int(self.data.shape[i]) for i, a in enumerate(self.axes)}
        lhs_resolved = [self._resolve_token_sizes(t, size_map) for t in lhs_resolved]

        if len(lhs_resolved) != len(self.axes):
            raise AxiomShapeError("LHS logical axes do not match physical tensor rank.")

        current_data = self.data
        unpacked_shape = []
        flat_lhs_tokens = []

        for i, (ax_def, token) in enumerate(zip(self.axes, lhs_resolved)):
            # The Ultimate Guardrail: LHS must exactly match physical layout
            token_name = getattr(token, "source_name", getattr(token, "name", str(token)))
            if token_name != ax_def.name:
                raise AxiomShapeError(
                    f"LHS axis mismatch at index {i}: expected physical axis '{ax_def.name}', "
                    f"but got '{token_name}'. The LHS must exactly describe the current physical layout. "
                    f"Use '->' for routing or reordering."
                )

            if isinstance(token, PackedAxis):
                token_name = getattr(token, "source_name", token.name)
                if token_name != ax_def.name:
                    raise AxiomShapeError(
                        f"Expected packed axis '{ax_def.name}', got '{token_name}'. Use '->' to route."
                    )

                if token.ops:
                    unpacked_shape.append(int(current_data.shape[i]))
                    flat_lhs_tokens.append(token)
                    continue

                unknown_count = sum(1 for a in token.axes if a.size is None)
                if unknown_count > 1:
                    unpacked_shape.append(int(current_data.shape[i]))
                    flat_lhs_tokens.append(token)
                else:
                    for a in token.axes:
                        unpacked_shape.append(a.size if a.size is not None else -1)
                        flat_lhs_tokens.append(a)
            else:
                unpacked_shape.append(int(current_data.shape[i]))
                flat_lhs_tokens.append(token)

        current_data = jnp.reshape(current_data, unpacked_shape)
        current_axis_names = [t.name for t in flat_lhs_tokens]
        self._assert_unique_names(current_axis_names, "LHS after unpack")

        surviving_tokens = []
        for token in flat_lhs_tokens:
            if isinstance(token, (Axis, PackedAxis)):
                idx = current_axis_names.index(token.name)
                current_data, current_axis_names, current_token = self._execute_ops(
                    current_data, current_axis_names, token, idx
                )
                self._assert_unique_names(current_axis_names, f"after executing ops on '{token.name}'")
                surviving_tokens.append(current_token)

            elif isinstance(token, ConsumedSlot):
                idx = current_axis_names.index(token.name)

                if token.op == "sum":
                    current_data = jnp.sum(current_data, axis=idx)
                elif token.op == "mean":
                    current_data = jnp.mean(current_data, axis=idx)
                elif token.op == "max":
                    current_data = jnp.max(current_data, axis=idx)
                elif token.op == "min":
                    current_data = jnp.min(current_data, axis=idx)
                elif token.op == "var":
                    current_data = jnp.var(current_data, axis=idx)
                elif token.op == "std":
                    current_data = jnp.std(current_data, axis=idx)

                # --- Phase 2: Advanced Reductions ---
                elif token.op == "logsumexp":
                    current_data = jsp.special.logsumexp(current_data, axis=idx)
                elif token.op == "argmax":
                    current_data = jnp.argmax(current_data, axis=idx)
                elif token.op == "argmin":
                    current_data = jnp.argmin(current_data, axis=idx)
                elif token.op == "any":
                    current_data = jnp.any(current_data, axis=idx)
                elif token.op == "all":
                    current_data = jnp.all(current_data, axis=idx)
                else:
                    raise AxiomSyntaxError(f"Unknown reduction op '{token.op}'.")
                current_axis_names.pop(idx)
                self._assert_unique_names(current_axis_names, f"after reduction '{token.op}'")
            else:
                raise AxiomSyntaxError(f"Unsupported token type on LHS: {type(token)}")

        if rhs_tokens is not None:
            rhs_resolved = self._resolve_ellipsis(rhs_tokens, tuple(surviving_tokens))

            # Resolve RHS symbolic sizes using the state immediately before layout execution!
            size_map = {name: int(current_data.shape[i]) for i, name in enumerate(current_axis_names)}
            rhs_resolved = [self._resolve_token_sizes(t, size_map) for t in rhs_resolved]

            current_data, routing_tokens = self._normalize_surviving_for_rhs(current_data, surviving_tokens)
            lhs_names = [t.name for t in routing_tokens]
            self._assert_unique_names(lhs_names, "RHS source axes")

            flat_rhs_names = []
            for t in rhs_resolved:
                if isinstance(t, PackedAxis):
                    if t.name in lhs_names:
                        flat_rhs_names.append(t.name)
                    else:
                        flat_rhs_names.extend([a.name for a in t.axes])
                else:
                    flat_rhs_names.append(t.name)

            self._assert_unique_names(flat_rhs_names, "RHS flat routing request")

            try:
                perm = [lhs_names.index(name) for name in flat_rhs_names]
            except ValueError as e:
                raise AxiomShapeError(
                    f"RHS references axes {flat_rhs_names}, but available routed axes are {lhs_names}."
                ) from e

            current_data = jnp.transpose(current_data, axes=perm)

            target_shape = []
            for token in rhs_resolved:
                if isinstance(token, PackedAxis):
                    if token.name in lhs_names:
                        idx = flat_rhs_names.index(token.name)
                        target_shape.append(int(current_data.shape[idx]))
                    else:
                        size = 1
                        for sub_ax in token.axes:
                            idx = flat_rhs_names.index(sub_ax.name)
                            size *= int(current_data.shape[idx])
                        target_shape.append(size)
                else:
                    idx = flat_rhs_names.index(token.name)
                    target_shape.append(int(current_data.shape[idx]))

            final_data = jnp.reshape(current_data, target_shape)
            final_axis_names = [t.name for t in rhs_resolved]
            self._assert_unique_names(final_axis_names, "RHS after reshape")

            final_tokens = []
            for idx, token in enumerate(rhs_resolved):
                if hasattr(token, "ops") and token.ops:
                    final_data, final_axis_names, final_token = self._execute_ops(
                        final_data, final_axis_names, token, idx
                    )
                    self._assert_unique_names(final_axis_names, f"after RHS ops on '{token.name}'")
                else:
                    if isinstance(token, PackedAxis):
                        final_token = PackedAxis(*[Axis(a.name, a.size, source_name=a.name) for a in token.axes])
                        if hasattr(token, "source_name"):
                            final_token.source_name = token.source_name
                    else:
                        final_token = Axis(token.name, int(final_data.shape[idx]), source_name=token.name)
                final_tokens.append(final_token)

            return AxiomTensor(final_data, self._finalize_output_axes(final_tokens, final_data))

        return AxiomTensor(current_data, self._finalize_output_axes(surviving_tokens, current_data))


class AxiomDropout(nnx.Module):
    """A truly random, mathematically pure, remat-safe Dropout layer."""

    def __init__(self, rate: float):
        self.rate = rate
        self.deterministic = False  # Standard NNX toggle pattern
        self.layer_id = id(self) & 0xFFFFFFFF

    def __call__(self, x):
        if self.rate == 0.0 or self.deterministic:
            return x

        step_key = context.step_key
        if step_key is None:
            step_key = jax.random.key(0)  # Safety fallback

        # Fold in the layer's unique ID so no two layers share the same mask
        key = jax.random.fold_in(step_key, self.layer_id)

        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))