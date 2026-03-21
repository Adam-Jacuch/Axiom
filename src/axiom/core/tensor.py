import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Any, List
from .axis import Axis, ConsumedSlot, PackedAxis, ProjOp, BiasOp, GateOp, NormOp, DropoutOp, ScanOp, ConvOp, CastOp
from .module import context
from .. import init as a_init
from ..exceptions import AxiomSyntaxError, AxiomShapeError


class AxiomTensor:
    def __init__(self, data: jnp.ndarray, logical_axes: Tuple[Axis, ...]):
        self.data = data
        self.axes = logical_axes

    def rename(self, old_axis, new_axis):
        """Explicitly renames a logical axis to maintain structural safety."""
        # Swap out the old axis for the new one while keeping the exact same physical data
        new_axes = tuple(new_axis if a.name == old_axis.name else a for a in self.axes)
        return AxiomTensor(self.data, new_axes)

    def __add__(self, other):
        if isinstance(other, AxiomTensor):
            self_names = [a.name for a in self.axes]
            other_names = [a.name for a in other.axes]
            if self_names != other_names:
                raise AxiomShapeError(f"Cannot add tensors with mismatched axes: {self_names} vs {other_names}")
            return AxiomTensor(self.data + other.data, self.axes)
        # Allows adding scalars (like x + 1)
        return AxiomTensor(self.data + other, self.axes)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, AxiomTensor):
            self_names = [a.name for a in self.axes]
            other_names = [a.name for a in other.axes]
            if self_names != other_names:
                raise AxiomShapeError(f"Cannot multiply tensors with mismatched axes: {self_names} vs {other_names}")
            return AxiomTensor(self.data * other.data, self.axes)
        # Allows scalar multiplication (like x * 0.5)
        return AxiomTensor(self.data * other, self.axes)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, AxiomTensor):
            self_names = [a.name for a in self.axes]
            other_names = [a.name for a in other.axes]
            if self_names != other_names:
                raise AxiomShapeError(f"Cannot divide tensors with mismatched axes: {self_names} vs {other_names}")
            return AxiomTensor(self.data / other.data, self.axes)
        return AxiomTensor(self.data / other, self.axes)

    def __rtruediv__(self, other):
        # Handles cases like `1.0 / tensor`
        return AxiomTensor(other / self.data, self.axes)

    def __sub__(self, other):
        if isinstance(other, AxiomTensor):
            self_names = [a.name for a in self.axes]
            other_names = [a.name for a in other.axes]
            if self_names != other_names:
                raise AxiomShapeError(f"Cannot subtract tensors with mismatched axes: {self_names} vs {other_names}")
            return AxiomTensor(self.data - other.data, self.axes)
        return AxiomTensor(self.data - other, self.axes)

    def _resolve_ellipsis(self, tokens: Tuple[Any, ...], reference_axes: Tuple[Axis, ...]) -> List[Any]:
        if Ellipsis not in tokens: return list(tokens)
        if tokens.count(Ellipsis) > 1: raise AxiomSyntaxError("Only one Ellipsis (...) is allowed per side.")

        explicit_names = set()
        for t in tokens:
            if t is Ellipsis: continue
            if hasattr(t, 'source_name'):
                # If the PackedAxis was mapped via '>>', use its top-level source!
                explicit_names.add(t.source_name)
            elif hasattr(t, 'axes'):
                # Otherwise, it's a standard PackedAxis, so look at its children
                explicit_names.update([getattr(a, 'source_name', a.name) for a in t.axes])
                # Also register the unified parent name! (e.g., 'h&dh')
                explicit_names.add(t.name)
            else:
                explicit_names.add(getattr(t, 'source_name', t.name))

        hidden_axes = [ax for ax in reference_axes if ax.name not in explicit_names]
        resolved = []
        for t in tokens:
            if t is Ellipsis:
                resolved.extend([Axis(a.name, a.size) for a in hidden_axes])
            else:
                resolved.append(t)
        return resolved

    def _execute_ops(self, current_data, current_axis_names, token, idx):
        """Helper to execute the pipeline operations for a specific axis."""
        current_name = token.name
        current_size = token.size

        for op in token.ops:
            # Activations
            if op == 'relu':
                current_data = jax.nn.relu(current_data)
            elif op == 'silu':
                current_data = jax.nn.silu(current_data)
            elif op == 'gelu':
                current_data = jax.nn.gelu(current_data)
            elif op == 'sigmoid':
                current_data = jax.nn.sigmoid(current_data)
            elif op == 'tanh':
                current_data = jnp.tanh(current_data)
            elif op == 'softmax':
                current_data = jax.nn.softmax(current_data, axis=idx)

            # Casting
            elif isinstance(op, CastOp):
                current_data = current_data.astype(op.dtype)

            # Masking
            elif isinstance(op, str) and op.startswith('mask_'):
                try:
                    sq_idx = current_axis_names.index('sq')
                except ValueError:
                    sq_idx = idx - 1
                sq_len, sk_len = current_data.shape[sq_idx], current_data.shape[idx]
                row_idx = jnp.arange(sq_len).reshape([-1 if i == sq_idx else 1 for i in range(current_data.ndim)])
                col_idx = jnp.arange(sk_len).reshape([-1 if i == idx else 1 for i in range(current_data.ndim)])
                mask = row_idx >= col_idx
                current_data = jnp.where(mask, current_data, -1e9)

            # Element-wise Gating
            elif isinstance(op, GateOp):
                current_data = current_data * op.gate_tensor.data

            # Implicit Bias
            elif isinstance(op, BiasOp):
                in_features = current_data.shape[idx]
                active_mod = context.get_active()
                if active_mod is None: raise RuntimeError("Implicit .bias() must be inside an axiom.Module.")
                param_name = f"_axiom_bias_{active_mod._axiom_param_counter}"
                active_mod._axiom_param_counter += 1

                if not getattr(active_mod, '_axiom_initialized', False):
                    b_init = op.init_fn if op.init_fn is not None else a_init.zeros
                    rng = nnx.Rngs(0)()
                    setattr(active_mod, param_name, nnx.Param(b_init(rng, (in_features,))))

                bias_param = getattr(active_mod, param_name)
                broadcast_shape = [1] * current_data.ndim
                broadcast_shape[idx] = in_features
                current_data = current_data + jnp.reshape(bias_param.value, broadcast_shape)

            # Normalization
            elif isinstance(op, NormOp):
                in_features = current_data.shape[idx]
                active_mod = context.get_active()
                if active_mod is None: raise RuntimeError("Implicit norm must be inside an axiom.Module.")
                param_name = f"_axiom_norm_{active_mod._axiom_param_counter}"
                active_mod._axiom_param_counter += 1

                if not getattr(active_mod, '_axiom_initialized', False):
                    if op.norm_type == 'rms':
                        i_scale = op.init_scale if op.init_scale is not None else a_init.ones
                        setattr(active_mod, param_name,
                                nnx.RMSNorm(in_features, epsilon=op.eps, scale_init=i_scale, rngs=nnx.Rngs(0)))
                    else:
                        setattr(active_mod, param_name,
                                nnx.LayerNorm(in_features, epsilon=op.eps, use_bias=op.use_bias, use_scale=op.use_scale,
                                              rngs=nnx.Rngs(0)))

                norm_layer = getattr(active_mod, param_name)
                current_data = jnp.swapaxes(current_data, idx, -1)
                current_data = norm_layer(current_data)
                current_data = jnp.swapaxes(current_data, -1, idx)

            # Dropout
            elif isinstance(op, DropoutOp):
                active_mod = context.get_active()
                if active_mod is None: raise RuntimeError("Dropout must be inside an axiom.Module.")
                param_name = f"_axiom_drop_{active_mod._axiom_param_counter}"
                active_mod._axiom_param_counter += 1

                if not getattr(active_mod, '_axiom_initialized', False):
                    setattr(active_mod, param_name, nnx.Dropout(rate=op.rate, rngs=nnx.Rngs(dropout=0)))

                drop_layer = getattr(active_mod, param_name)
                current_data = drop_layer(current_data)

            # Recurrent Scan
            elif isinstance(op, ScanOp):
                scan_main = jnp.swapaxes(current_data, 0, idx)
                scan_extras = []
                for extra in op.inputs:
                    e_names = [getattr(a, 'source_name', a.name) for a in extra.axes]
                    e_idx = e_names.index(token.source_name)
                    scan_extras.append(jnp.swapaxes(extra.data, 0, e_idx))

                init_state = jnp.zeros_like(scan_main[0])

                def scan_body(carry, xs):
                    new_state, out_y = op.fn(carry, (xs[0], *xs[1:]))
                    new_state = new_state.astype(carry.dtype)
                    return new_state, out_y

                _, scanned_out = jax.lax.scan(scan_body, init_state, (scan_main, *scan_extras))
                current_data = jnp.swapaxes(scanned_out, 0, idx)

            # Projection & Contraction
            elif isinstance(op, ProjOp):
                if op.weight is None:
                    in_features = current_data.shape[idx]

                    if isinstance(op.out_axis, PackedAxis):
                        known_size = 1
                        unknowns = []
                        for a in op.out_axis.axes:
                            if a.size is not None:
                                known_size *= a.size
                            else:
                                unknowns.append(a)

                        if len(unknowns) == 1:
                            unknowns[0].size = in_features // known_size
                            out_features = in_features
                        else:
                            out_features = known_size
                    else:
                        out_features = op.out_axis.size if op.out_axis.size is not None else in_features
                        op.out_axis.size = out_features

                    active_mod = context.get_active()
                    if active_mod is None: raise RuntimeError("Implicit .proj() must be inside an axiom.Module.")
                    param_name = f"_axiom_proj_{active_mod._axiom_param_counter}"
                    active_mod._axiom_param_counter += 1

                    if not getattr(active_mod, '_axiom_initialized', False):
                        k_init = op.kernel_init if op.kernel_init is not None else a_init.default_kernel_init
                        b_init = op.bias_init if op.bias_init is not None else a_init.default_bias_init
                        p_dtype = op.dtype if op.dtype is not None else jnp.float32
                        setattr(active_mod, param_name, nnx.Linear(
                            in_features, out_features, use_bias=op.use_bias,
                            kernel_init=k_init, bias_init=b_init,
                            dtype=p_dtype, param_dtype=p_dtype,
                            rngs=nnx.Rngs(0)
                        ))

                    linear_layer = getattr(active_mod, param_name)
                    current_data = jnp.swapaxes(current_data, idx, -1)
                    current_data = linear_layer(current_data)
                    current_data = jnp.swapaxes(current_data, -1, idx)

                    current_axis_names[idx] = op.out_axis.name
                    current_name = op.out_axis.name
                    current_size = out_features

                else:
                    weight_tensor = op.weight
                    lhs_names = current_axis_names.copy()
                    rhs_names = [getattr(a, 'source_name', a.name) for a in weight_tensor.axes]
                    out_names = [op.out_axis.name if n == token.source_name else n for n in lhs_names]

                    unique_names = list(set(lhs_names + rhs_names + out_names))
                    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    n2c = {name: chars[i] for i, name in enumerate(unique_names)}

                    lhs_str = "".join(n2c[n] for n in lhs_names)
                    rhs_str = "".join(n2c[n] for n in rhs_names)
                    out_str = "".join(n2c[n] for n in out_names)

                    current_data = jnp.einsum(f"{lhs_str},{rhs_str}->{out_str}", current_data, weight_tensor.data)
                    current_axis_names = out_names
                    current_name = op.out_axis.name
                    idx = current_axis_names.index(current_name)
                    current_size = current_data.shape[idx]
            elif isinstance(op, ConvOp):
                in_features = current_data.shape[idx]
                out_features = op.features.size
                out_name = op.features.name

                active_mod = context.get_active()
                if active_mod is None: raise RuntimeError("Implicit .conv() must be inside an axiom.Module.")
                param_name = f"_axiom_conv_{active_mod._axiom_param_counter}"
                active_mod._axiom_param_counter += 1

                if not getattr(active_mod, '_axiom_initialized', False):
                    k_init = op.kernel_init if op.kernel_init is not None else a_init.default_kernel_init
                    b_init = op.bias_init if op.bias_init is not None else a_init.default_bias_init

                    setattr(active_mod, param_name, nnx.Conv(
                        in_features=in_features,
                        out_features=out_features,
                        kernel_size=op.kernel_size,
                        strides=op.strides,
                        padding=op.padding,
                        use_bias=op.use_bias,
                        kernel_init=k_init,
                        bias_init=b_init,
                        rngs=nnx.Rngs(0)
                    ))

                conv_layer = getattr(active_mod, param_name)
                # Swap channel axis to the end so nnx.Conv can process spatial dimensions correctly
                current_data = jnp.swapaxes(current_data, idx, -1)
                current_data = conv_layer(current_data)
                current_data = jnp.swapaxes(current_data, -1, idx)

                current_axis_names[idx] = out_name
                current_name = out_name
                current_size = out_features

        return current_data, current_axis_names, current_name, current_size

    def __getitem__(self, tokens: Any) -> 'AxiomTensor':
        if not isinstance(tokens, tuple): tokens = (tokens,)
        arrow_count = tokens.count("->")
        if arrow_count > 1: raise AxiomSyntaxError("Multiple '->' operators found.")

        if arrow_count == 1:
            split_idx = tokens.index("->")
            lhs_tokens, rhs_tokens = tokens[:split_idx], tokens[split_idx + 1:]
        else:
            lhs_tokens, rhs_tokens = tokens, None

        lhs_resolved = self._resolve_ellipsis(lhs_tokens, self.axes)

        # --- 1. Smart LHS Alignment & Unpacking ---
        if len(lhs_resolved) != len(self.axes):
            raise AxiomShapeError(f"LHS logical axes do not match physical tensor rank.")

        current_data = self.data
        unpacked_shape = []
        flat_lhs_tokens = []

        for ax_def, token in zip(self.axes, lhs_resolved):
            if isinstance(token, PackedAxis):
                # THE FIX: Check source_name first! If it's an alias, this will match ax_def.
                token_name = getattr(token, 'source_name', token.name)

                if token_name != ax_def.name:
                    raise AxiomShapeError(
                        f"Expected packed axis '{ax_def.name}', got '{token_name}'. Use '->' to route.")

                # Dynamic Unpack Check
                unknown_count = sum(1 for a in token.axes if a.size is None)
                if unknown_count > 1:
                    # Too many unknowns. Treat as a single, unified physical block.
                    unpacked_shape.append(current_data.shape[self.axes.index(ax_def)])
                    flat_lhs_tokens.append(token)
                else:
                    # Safe to unpack! Let JAX infer the single missing dimension.
                    for a in token.axes:
                        unpacked_shape.append(a.size if a.size is not None else -1)
                        flat_lhs_tokens.append(a)
            else:
                unpacked_shape.append(current_data.shape[self.axes.index(ax_def)])
                flat_lhs_tokens.append(token)

        current_data = jnp.reshape(current_data, unpacked_shape)
        current_axis_names = [getattr(a, 'source_name', a.name) for a in flat_lhs_tokens]

        # --- 2. LHS Pipeline Execution ---
        surviving_axes = []
        for token in flat_lhs_tokens:
            if isinstance(token, (Axis, PackedAxis)):
                idx = current_axis_names.index(getattr(token, 'source_name', token.name))
                current_data, current_axis_names, current_name, current_size = self._execute_ops(
                    current_data, current_axis_names, token, idx
                )
                surviving_axes.append(Axis(current_name, current_size))

            elif isinstance(token, ConsumedSlot):
                idx = current_axis_names.index(token.source_name)
                if token.op == 'sum':
                    current_data = jnp.sum(current_data, axis=idx)
                elif token.op == 'mean':
                    current_data = jnp.mean(current_data, axis=idx)
                elif token.op == 'max':
                    current_data = jnp.max(current_data, axis=idx)
                elif token.op == 'min':
                    current_data = jnp.min(current_data, axis=idx)
                elif token.op == 'var':
                    current_data = jnp.var(current_data, axis=idx)
                elif token.op == 'std':
                    current_data = jnp.std(current_data, axis=idx)
                current_axis_names.pop(idx)

        # --- 3. Smart RHS Routing ---
        if rhs_tokens is not None:
            rhs_resolved = self._resolve_ellipsis(rhs_tokens, tuple(surviving_axes))

            flat_rhs_names = []
            lhs_names = [a.name for a in surviving_axes]

            for t in rhs_resolved:
                if isinstance(t, PackedAxis):
                    # If LHS already has this axis unified, keep it unified!
                    if t.name in lhs_names:
                        flat_rhs_names.append(t.name)
                    else:
                        flat_rhs_names.extend([a.name for a in t.axes])
                else:
                    flat_rhs_names.append(t.name)

            perm = [lhs_names.index(name) for name in flat_rhs_names]
            current_data = jnp.transpose(current_data, axes=perm)

            target_shape = []
            for token in rhs_resolved:
                if isinstance(token, PackedAxis):
                    if token.name in lhs_names:
                        idx = flat_rhs_names.index(token.name)
                        target_shape.append(current_data.shape[idx])
                    else:
                        size = 1
                        for sub_ax in token.axes:
                            idx = flat_rhs_names.index(sub_ax.name)
                            size *= current_data.shape[idx]
                        target_shape.append(size)
                else:
                    idx = flat_rhs_names.index(token.name)
                    target_shape.append(current_data.shape[idx])

            final_data = jnp.reshape(current_data, target_shape)
            final_axis_names = [t.name for t in rhs_resolved]

            # --- 4. RHS Pipeline Execution ---
            surviving_final_axes = []
            for idx, token in enumerate(rhs_resolved):
                if hasattr(token, 'ops') and token.ops:
                    final_data, final_axis_names, final_name, final_size = self._execute_ops(
                        final_data, final_axis_names, token, idx
                    )
                else:
                    final_name = token.name
                    final_size = final_data.shape[idx]

                surviving_final_axes.append(Axis(final_name, final_size))

            return AxiomTensor(final_data, tuple(surviving_final_axes))

        else:
            return AxiomTensor(current_data, tuple(surviving_axes))