import unittest

import jax
import jax.numpy as jnp
import numpy as np

from axiom import ax, tensor, Module
from axiom.exceptions import AxiomShapeError, AxiomSyntaxError
from axiom.core.axis import SymbolicSize

import jax.scipy as jsp


def const_init(value, dtype=jnp.float32):
    def _init(rng, shape, param_dtype=None):
        use_dtype = dtype if param_dtype is None else param_dtype
        return jnp.full(shape, value, dtype=use_dtype)
    return _init


def assert_axes(testcase: unittest.TestCase, t, names, sizes=None):
    got_names = [a.name for a in t.axes]
    testcase.assertEqual(got_names, list(names))
    if sizes is not None:
        got_sizes = [a.size for a in t.axes]
        testcase.assertEqual(got_sizes, list(sizes))


def assert_allclose(x, y, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=rtol)


class ImplicitCausalConv1DMod(Module):
    def __call__(self, x):
        return x[..., ax.c.conv(
            3,
            over=ax.s.causal(),
            out=ax.cout(1),
            use_bias=False,
            kernel_init=const_init(1.0),
        )]


class ProjDefaultBiasMod(Module):
    def __call__(self, x):
        return x[..., ax.d.proj(
            out=ax.o(3),
            kernel_init=const_init(0.0),
            bias_init=const_init(2.0),
        )]


class ProjNoBiasMod(Module):
    def __call__(self, x):
        return x[..., ax.d.proj(
            out=ax.o(3),
            use_bias=False,
            kernel_init=const_init(0.0),
            bias_init=const_init(2.0),
        )]


class BiasImplicitMod(Module):
    def __call__(self, x):
        return x[..., ax.d.bias(init_fn=const_init(2.0))]


class GateImplicitMod(Module):
    def __call__(self, x):
        return x[..., ax.d.gate(init_fn=const_init(3.0))]


class RMSImplicitScaleMod(Module):
    def __call__(self, x):
        return x[..., ax.d.norm_rms(init_scale=const_init(2.0))]


class DropoutZeroMod(Module):
    def __call__(self, x):
        return x[..., ax.d.dropout(0.0)]


class EmbedMod(Module):
    def __call__(self, x):
        return x.embed(vocab=5, out=ax.d(3), return_weight=True)


class AxisScaledProjMod(Module):
    def __call__(self, x):
        # Notice we don't need to define the size! ax.d * 2 is fully symbolic now.
        return x[..., ax.d.proj(out=ax.d2(ax.d * 2), use_bias=False, kernel_init=const_init(0.0))]


class AxisFloorDivProjMod(Module):
    def __call__(self, x):
        # Fully symbolic lazy evaluation
        return x[..., ax.d.proj(out=ax.d2(ax.d // 2), use_bias=False, kernel_init=const_init(0.0))]


class AxiomRuntimeTests(unittest.TestCase):
    def test_axiomtensor_is_pytree_and_tree_map_roundtrips(self):
        x = tensor(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), ax.b, ax.d)
        y = jax.tree_util.tree_map(lambda arr: arr + 1, x)

        self.assertEqual(type(y), type(x))
        assert_axes(self, y, ["b", "d"], [None, None])
        assert_allclose(y.data, x.data + 1)

    def test_named_transpose_pack_unpack(self):
        x = tensor(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), ax.b, ax.s, ax.d)

        y = x[ax.b, ax.s, ax.d, "->", ax.b, ax.d, ax.s]
        assert_axes(self, y, ["b", "d", "s"], [2, 4, 3])
        assert_allclose(y.data, jnp.transpose(x.data, (0, 2, 1)))

        p = y[ax.b, ax.d, ax.s, "->", ax.b, ax.d & ax.s]
        assert_axes(self, p, ["b", "d&s"], [2, 12])

        z = p[ax.b, ax.d(4) & ax.s(3), "->", ax.b, ax.d, ax.s]
        assert_axes(self, z, ["b", "d", "s"], [2, 4, 3])
        assert_allclose(z.data, y.data)

    def test_ellipsis_route(self):
        x = tensor(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), ax.b, ax.s, ax.d)
        y = x[..., ax.d, "->", ax.d, ...]
        assert_axes(self, y, ["d", "b", "s"], [4, 2, 3])
        assert_allclose(y.data, jnp.transpose(x.data, (2, 0, 1)))

    def test_multiple_arrow_raises(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)
        with self.assertRaises(AxiomSyntaxError):
            _ = x[ax.b, "->", ax.d, "->", ax.b]

    def test_multiple_ellipsis_raises(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)
        with self.assertRaises(AxiomSyntaxError):
            _ = x[..., ..., ax.d]

    def test_axiomtensor_arithmetic_is_strict_but_raw_arrays_broadcast(self):
        x = tensor(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), ax.b, ax.d)
        y = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)
        z = x + y
        assert_axes(self, z, ["b", "d"], [None, None])
        assert_allclose(z.data, x.data + y.data)

        y_bad = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.d, ax.b)
        with self.assertRaises(AxiomShapeError):
            _ = x + y_bad

        raw = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
        z2 = x + raw
        assert_axes(self, z2, ["b", "d"], [None, None])
        assert_allclose(z2.data, x.data + raw)

    # -------------------------------------------------------------------------
    # Axis arithmetic (Now Fully Symbolic!)
    # -------------------------------------------------------------------------

    def test_axis_mul_known_size(self):
        d = ax.d(8)
        d2 = d * 2
        self.assertEqual(d2.name, "d")
        self.assertEqual(d2.size, 16)
        self.assertEqual(d2.source_name, d.source_name)

    def test_axis_rmul_known_size(self):
        d = ax.d(8)
        d2 = 3 * d
        self.assertEqual(d2.name, "d")
        self.assertEqual(d2.size, 24)

    def test_axis_floordiv_known_size(self):
        d = ax.d(12)
        d2 = d // 3
        self.assertEqual(d2.name, "d")
        self.assertEqual(d2.size, 4)

    def test_axis_mul_unknown_size_creates_symbolic_promise(self):
        d2 = ax.d * 2
        self.assertEqual(d2.name, "d")
        self.assertTrue(isinstance(d2.size, SymbolicSize))
        self.assertEqual(d2.size.op, "*")
        self.assertEqual(d2.size.value, 2)

    def test_axis_floordiv_unknown_size_creates_symbolic_promise(self):
        d2 = ax.d // 2
        self.assertTrue(isinstance(d2.size, SymbolicSize))
        self.assertEqual(d2.size.op, "//")
        self.assertEqual(d2.size.value, 2)

    def test_axis_floordiv_non_divisible_raises(self):
        with self.assertRaises(AxiomShapeError):
            _ = ax.d(10) // 3

    def test_axis_scaled_projection_resolves_symbolic_size(self):
        x = tensor(jnp.ones((2, 4), dtype=jnp.float32), ax.b, ax.d)
        mod = AxisScaledProjMod()
        y = mod(x)

        assert_axes(self, y, ["b", "d2"], [2, 8])
        assert_allclose(y.data, jnp.zeros((2, 8), dtype=jnp.float32))

    def test_axis_floordiv_projection_resolves_symbolic_size(self):
        x = tensor(jnp.ones((2, 8), dtype=jnp.float32), ax.b, ax.d)
        mod = AxisFloorDivProjMod()
        y = mod(x)

        assert_axes(self, y, ["b", "d2"], [2, 4])
        assert_allclose(y.data, jnp.zeros((2, 4), dtype=jnp.float32))

    # -------------------------------------------------------------------------
    # Positional Indexing (.idx)
    # -------------------------------------------------------------------------

    def test_idx_integer_drops_axis(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3), ax.b, ax.s, ax.d)
        y = x.idx[0]
        assert_axes(self, y, ["s", "d"], [2, 3])
        assert_allclose(y.data, x.data[0])

    def test_idx_slice_updates_size(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 6), ax.b, ax.s)
        y = x.idx[:, 1:4]
        assert_axes(self, y, ["b", "s"], [2, 3])
        assert_allclose(y.data, x.data[:, 1:4])

    def test_idx_ellipsis(self):
        x = tensor(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), ax.b, ax.s, ax.d)
        y = x.idx[..., 1]
        assert_axes(self, y, ["b", "s"], [2, 3])
        assert_allclose(y.data, x.data[..., 1])

    def test_idx_errors(self):
        x = tensor(jnp.arange(6).reshape(2, 3), ax.b, ax.d)
        with self.assertRaises(AxiomShapeError):
            _ = x.idx[None]  # no newaxis supported
        with self.assertRaises(AxiomSyntaxError):
            _ = x.idx[..., ...]  # double ellipsis
        with self.assertRaises(AxiomShapeError):
            _ = x.idx[0, 0, 0]  # too many dims

    # -------------------------------------------------------------------------
    # Remaining Tests
    # -------------------------------------------------------------------------

    def test_duplicate_axis_names_rejected_on_projection(self):
        x = tensor(jnp.ones((2, 4, 8), dtype=jnp.float32), ax.b, ax.sq, ax.d)
        with self.assertRaises(AxiomShapeError):
            _ = x[..., ax.d.proj(out=ax.sq(8), use_bias=False)]

    def test_duplicate_axis_names_rejected_on_alias(self):
        x = tensor(jnp.ones((2, 4, 5), dtype=jnp.float32), ax.b, ax.sq, ax.sk)
        with self.assertRaises(AxiomShapeError):
            _ = x[ax.b, (ax.sq >> ax.sk), ax.sk]

    def test_proj_default_use_bias_true(self):
        x = tensor(jnp.zeros((2, 4), dtype=jnp.float32), ax.b, ax.d)
        mod = ProjDefaultBiasMod()
        y = mod(x)

        assert_axes(self, y, ["b", "o"], [2, 3])
        assert_allclose(y.data, jnp.full((2, 3), 2.0, dtype=jnp.float32))

    def test_proj_use_bias_false(self):
        x = tensor(jnp.zeros((2, 4), dtype=jnp.float32), ax.b, ax.d)
        mod = ProjNoBiasMod()
        y = mod(x)

        assert_axes(self, y, ["b", "o"], [2, 3])
        assert_allclose(y.data, jnp.zeros((2, 3), dtype=jnp.float32))

    def test_explicit_weighted_proj_with_explicit_bias(self):
        x = tensor(jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32), ax.b, ax.d)
        w = tensor(
            jnp.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ], dtype=jnp.float32),
            ax.d,
            ax.o,
        )
        b = tensor(jnp.array([10.0, 20.0], dtype=jnp.float32), ax.o)

        y = x[..., ax.d.proj(out=ax.o(2), weight=w, bias=b, use_bias=False)]

        expected = jnp.einsum("bd,do->bo", x.data, w.data) + b.data
        assert_axes(self, y, ["b", "o"], [1, 2])
        assert_allclose(y.data, expected)

    def test_explicit_weighted_proj_to_packed_axis_raises(self):
        x = tensor(jnp.ones((2, 8), dtype=jnp.float32), ax.b, ax.d)
        w = tensor(jnp.ones((8, 2, 4), dtype=jnp.float32), ax.d, ax.h, ax.dh)

        with self.assertRaises(AxiomShapeError):
            _ = x[..., ax.d.proj(out=ax.h(2) & ax.dh(4), weight=w, use_bias=False)]

    def test_implicit_bias(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)
        mod = BiasImplicitMod()
        y = mod(x)

        assert_axes(self, y, ["b", "d"], [2, 3])
        assert_allclose(y.data, x.data + 2.0)

    def test_explicit_bias_named_broadcast(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3), ax.b, ax.s, ax.d)
        b = tensor(jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32), ax.d)

        y = x[..., ax.d.bias(tensor=b)]
        expected = x.data + jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)

        assert_axes(self, y, ["b", "s", "d"], [2, 2, 3])
        assert_allclose(y.data, expected)

    def test_implicit_gate(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)
        mod = GateImplicitMod()
        y = mod(x)

        assert_axes(self, y, ["b", "d"], [2, 3])
        assert_allclose(y.data, x.data * 3.0)

    def test_explicit_gate_named_broadcast(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3), ax.b, ax.s, ax.d)
        g = tensor(jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32), ax.d)

        y = x[..., ax.d.gate(tensor=g)]
        expected = x.data * jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)

        assert_axes(self, y, ["b", "s", "d"], [2, 2, 3])
        assert_allclose(y.data, expected)

    def test_mask_tril_uses_left_axis_plane(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(3, 4), ax.sq, ax.sk)
        y = x[..., ax.sk.mask("tril")]

        row = jnp.arange(3)[:, None]
        col = jnp.arange(4)[None, :]
        fill = jnp.finfo(jnp.float32).min
        expected = jnp.where(row >= col, x.data, fill)

        assert_axes(self, y, ["sq", "sk"], [3, 4])
        assert_allclose(y.data, expected)

    def test_mask_triu_uses_left_axis_plane(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(3, 4), ax.sq, ax.sk)
        y = x[..., ax.sk.mask("triu")]

        row = jnp.arange(3)[:, None]
        col = jnp.arange(4)[None, :]
        fill = jnp.finfo(jnp.float32).min
        expected = jnp.where(row <= col, x.data, fill)

        assert_axes(self, y, ["sq", "sk"], [3, 4])
        assert_allclose(y.data, expected)

    def test_alias_visible_name_explicit_bias_works(self):
        x = tensor(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), ax.b, ax.sq)
        b = tensor(jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32), ax.sk)

        y = x[ax.b, (ax.sq >> ax.sk).bias(tensor=b)]

        assert_axes(self, y, ["b", "sk"], [2, 3])
        assert_allclose(y.data, x.data + jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32))

    def test_alias_visible_name_reduction_works(self):
        x = tensor(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), ax.b, ax.sq)
        y = x[ax.b, (ax.sq >> ax.sk).sum()]

        assert_axes(self, y, ["b"], [2])
        assert_allclose(y.data, jnp.sum(x.data, axis=1))

    def test_norm_rms_implicit_scale(self):
        x = tensor(jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32), ax.b, ax.d)
        mod = RMSImplicitScaleMod()
        y = mod(x)

        rms = jnp.sqrt(jnp.mean(x.data ** 2, axis=-1, keepdims=True) + 1e-5)
        expected = 2.0 * (x.data / rms)

        assert_axes(self, y, ["b", "d"], [1, 3])
        assert_allclose(y.data, expected, atol=1e-4, rtol=1e-4)

    def test_norm_rms_explicit_scale(self):
        x = tensor(jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32), ax.b, ax.d)
        scale = tensor(jnp.array([1.0, 2.0, 4.0], dtype=jnp.float32), ax.d)

        y = x[..., ax.d.norm_rms(scale=scale)]

        rms = jnp.sqrt(jnp.mean(x.data ** 2, axis=-1, keepdims=True) + 1e-5)
        expected = (x.data / rms) * jnp.array([1.0, 2.0, 4.0], dtype=jnp.float32)

        assert_axes(self, y, ["b", "d"], [1, 3])
        assert_allclose(y.data, expected, atol=1e-4, rtol=1e-4)

    def test_norm_layer_explicit_scale_and_bias(self):
        x = tensor(jnp.array([[1.0, 2.0, 4.0]], dtype=jnp.float32), ax.b, ax.d)
        scale = tensor(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), ax.d)
        bias = tensor(jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32), ax.d)

        y = x[..., ax.d.norm_layer(scale=scale, bias=bias, use_scale=False, use_bias=False)]

        mean = jnp.mean(x.data, axis=-1, keepdims=True)
        var = jnp.var(x.data, axis=-1, keepdims=True)
        normed = (x.data - mean) / jnp.sqrt(var + 1e-5)
        expected = normed * scale.data + bias.data

        assert_axes(self, y, ["b", "d"], [1, 3])
        assert_allclose(y.data, expected, atol=1e-4, rtol=1e-4)

    def test_activations(self):
        x = tensor(jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32), ax.d)

        cases = [
            ("relu", jax.nn.relu(x.data)),
            ("silu", jax.nn.silu(x.data)),
            ("gelu", jax.nn.gelu(x.data)),
            ("sigmoid", jax.nn.sigmoid(x.data)),
            ("tanh", jnp.tanh(x.data)),
            ("softmax", jax.nn.softmax(x.data, axis=-1)),
            ("softplus", jax.nn.softplus(x.data)),
        ]

        for op_name, expected in cases:
            with self.subTest(op_name=op_name):
                token = getattr(ax.d, op_name)()
                y = x[token]
                assert_axes(self, y, ["d"], [3])
                assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_cast(self):
        x = tensor(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), ax.d)
        y = x[ax.d.cast(jnp.bfloat16)]

        assert_axes(self, y, ["d"], [3])
        self.assertEqual(y.data.dtype, jnp.bfloat16)

    def test_dropout_rate_zero_is_noop(self):
        x = tensor(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), ax.b, ax.d)
        mod = DropoutZeroMod()
        y = mod(x)

        assert_axes(self, y, ["b", "d"], [2, 3])
        assert_allclose(y.data, x.data)

    def test_scan_simple_cumsum(self):
        def step(carry, xs):
            x_t = xs[0]
            new = carry + x_t
            return new, new

        x = tensor(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), ax.s)
        y = x[ax.s.scan(step)]

        expected = jnp.cumsum(x.data, axis=0)
        assert_axes(self, y, ["s"], [4])
        assert_allclose(y.data, expected)

    def test_scan_with_extra_input(self):
        def step(carry, xs):
            x_t, u_t = xs
            new = carry + x_t + u_t
            return new, new

        x = tensor(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), ax.s)
        u = tensor(jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32), ax.s)

        y = x[ax.s.scan(step, inputs=(u,))]
        expected = jnp.cumsum(x.data + u.data, axis=0)

        assert_axes(self, y, ["s"], [4])
        assert_allclose(y.data, expected)

    def test_explicit_conv_1d_kernel_and_bias(self):
        x = tensor(jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1), ax.b, ax.s, ax.c)
        w = jnp.ones((1, 1, 1), dtype=jnp.float32)
        b = tensor(jnp.array([2.0], dtype=jnp.float32), ax.cout)

        y = x[..., ax.c.conv(
            features=ax.cout(1),
            kernel_size=(1,),
            weight=w,
            bias=b,
            use_bias=False,
        )]

        expected = x.data + 2.0
        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 1])
        assert_allclose(y.data, expected)

    def test_embed_implicit_then_explicit_weight_reuse(self):
        x_ids = tensor(jnp.array([[0, 1, 2], [2, 3, 4]], dtype=jnp.int32), ax.b, ax.sq)

        mod = EmbedMod()
        y1, w = mod(x_ids)
        y2 = x_ids.embed(weight=w)

        assert_axes(self, y1, ["b", "sq", "d"], [None, None, 3])
        assert_axes(self, w, ["vocab", "d"], [5, 3])
        assert_axes(self, y2, ["b", "sq", "d"], [None, None, 3])

        assert_allclose(y2.data, y1.data)

    def test_implicit_ops_require_module(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.d)

        cases = [
            lambda t: t[..., ax.d.bias()],
            lambda t: t[..., ax.d.gate()],
            lambda t: t[..., ax.d.norm_rms()],
            lambda t: t[..., ax.d.proj(out=ax.o(2))],
            lambda t: t[..., ax.d.dropout(0.0)],
        ]

        for fn in cases:
            with self.subTest(fn=fn):
                with self.assertRaises(RuntimeError):
                    _ = fn(x)

    def test_implicit_conv_requires_module(self):
        x = tensor(jnp.ones((1, 4, 1), dtype=jnp.float32), ax.b, ax.s, ax.c)
        with self.assertRaises(RuntimeError):
            _ = x[..., ax.c.conv(features=ax.cout(1), kernel_size=(1,))]

    def test_implicit_embed_requires_module(self):
        x_ids = tensor(jnp.array([[0, 1, 2]], dtype=jnp.int32), ax.b, ax.sq)
        with self.assertRaises(RuntimeError):
            _ = x_ids.embed(vocab=5, out=ax.d(3))

    def test_ambiguity_checks(self):
        b = tensor(jnp.array([1.0, 2.0], dtype=jnp.float32), ax.o)
        with self.assertRaises(ValueError):
            _ = ax.d.proj(out=ax.o(2), bias=b, use_bias=True)

        b2 = tensor(jnp.array([1.0], dtype=jnp.float32), ax.cout)
        with self.assertRaises(ValueError):
            _ = ax.c.conv(features=ax.cout(1), kernel_size=(1,), bias=b2, use_bias=True)

        s = tensor(jnp.array([1.0, 2.0], dtype=jnp.float32), ax.d)
        with self.assertRaises(ValueError):
            _ = ax.d.norm_layer(scale=s, use_scale=True)

        b3 = tensor(jnp.array([1.0, 2.0], dtype=jnp.float32), ax.d)
        with self.assertRaises(ValueError):
            _ = ax.d.norm_layer(bias=b3, use_bias=True)

    def test_packing_restrictions(self):
        with self.assertRaises(ValueError):
            _ = ax.h.relu() & ax.dh

        p = (ax.h & ax.dh).norm_rms()
        with self.assertRaises(ValueError):
            _ = p & ax.x

    def test_reductions(self):
        x = tensor(jnp.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=jnp.float32), ax.b, ax.d)

        cases = [
            ("sum", jnp.sum(x.data, axis=1)),
            ("mean", jnp.mean(x.data, axis=1)),
            ("max", jnp.max(x.data, axis=1)),
            ("min", jnp.min(x.data, axis=1)),
            ("var", jnp.var(x.data, axis=1)),
            ("std", jnp.std(x.data, axis=1)),
        ]

        for reduction, expected in cases:
            with self.subTest(reduction=reduction):
                token = getattr(ax.d, reduction)()
                y = x[ax.b, token]
                assert_axes(self, y, ["b"], [2])
                assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_packed_lhs_gate_survives_and_can_unpack_on_rhs(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 6), ax.b, ax.flat(6))

        y = x[
            ax.b,
            (ax.flat >> (ax.h(2) & ax.dh(3))).gate(tensor=2.0),
            "->",
            ax.b,
            ax.h,
            ax.dh,
        ]

        expected = (x.data * 2.0).reshape(2, 2, 3)
        assert_axes(self, y, ["b", "h", "dh"], [2, 2, 3])
        assert_allclose(y.data, expected)

    def test_packed_lhs_norm_survives_and_can_unpack_on_rhs(self):
        x = tensor(jnp.arange(12, dtype=jnp.float32).reshape(2, 6) + 1.0, ax.b, ax.flat(6))

        y = x[
            ax.b,
            (ax.flat >> (ax.h(2) & ax.dh(3))).norm_rms(scale=1.0),
            "->",
            ax.b,
            ax.h,
            ax.dh,
        ]

        flat = x.data
        rms = jnp.sqrt(jnp.mean(flat ** 2, axis=-1, keepdims=True) + 1e-5)
        expected = (flat / rms).reshape(2, 2, 3)

        assert_axes(self, y, ["b", "h", "dh"], [2, 2, 3])
        assert_allclose(y.data, expected, atol=1e-4, rtol=1e-4)

    # -------------------------------------------------------------------------
    # Axiom v2 Structural Operations
    # -------------------------------------------------------------------------

    def test_where_op(self):
        x = tensor(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32), ax.b, ax.s)
        mask = tensor(jnp.array([[True, False], [False, True]]), ax.b, ax.s)
        zeros = tensor(jnp.zeros((2, 2), dtype=jnp.float32), ax.b, ax.s)

        y = x[..., ax.s.where(condition=mask, false_tensor=zeros)]

        expected = jnp.array([[1.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)
        assert_axes(self, y, ["b", "s"], [2, 2])
        assert_allclose(y.data, expected)

    def test_pad_op(self):
        x = tensor(jnp.ones((2, 3), dtype=jnp.float32), ax.b, ax.s)
        # Pad the sequence dimension by 1 on the left, 2 on the right
        y = x[..., ax.s.pad((1, 2), mode="constant", value=0.0)]

        expected = jnp.array([
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        ], dtype=jnp.float32)

        assert_axes(self, y, ["b", "s"], [2, 6])
        assert_allclose(y.data, expected)

    def test_gather_op_moe_routing(self):
        # 3 experts, each with a hidden size of 4
        experts = tensor(jnp.arange(12, dtype=jnp.float32).reshape(3, 4), ax.e(3), ax.d(4))
        # Batch of 2 tokens, routed to expert 2 and expert 0
        routing_idx = tensor(jnp.array([2, 0], dtype=jnp.int32), ax.b(2))

        # Gather along the expert dimension
        routed = experts[ax.e.gather(routing_idx), ax.d]

        expected = jnp.array([
            [8.0, 9.0, 10.0, 11.0],  # Expert 2
            [0.0, 1.0, 2.0, 3.0]  # Expert 0
        ], dtype=jnp.float32)

        assert_axes(self, routed, ["b", "d"], [2, 4])
        assert_allclose(routed.data, expected)

    def test_roll_op_rope_shift(self):
        x = tensor(jnp.arange(4, dtype=jnp.float32), ax.d)
        y = x[ax.d.roll(shift=1)]

        expected = jnp.array([3.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
        assert_axes(self, y, ["d"], [4])
        assert_allclose(y.data, expected)

    def test_fill_op(self):
        x = tensor(jnp.arange(4, dtype=jnp.float32), ax.d)
        y = x[ax.d.fill(-1.0)]

        expected = jnp.full((4,), -1.0, dtype=jnp.float32)
        assert_axes(self, y, ["d"], [4])
        assert_allclose(y.data, expected)

    def test_flash_attention_routing(self):
        # [batch, heads, seq, head_dim]
        q = tensor(jnp.ones((2, 4, 8, 16), dtype=jnp.float32), ax.b, ax.h, ax.sq, ax.dh)

        # Notice we changed sq to sk for keys and values
        k = tensor(jnp.ones((2, 4, 8, 16), dtype=jnp.float32), ax.b, ax.h, ax.sk, ax.dh)
        v = tensor(jnp.ones((2, 4, 8, 16), dtype=jnp.float32), ax.b, ax.h, ax.sk, ax.dh)

        # FIXED: explicitly list ax.dh at the end so it matches the physical layout [b, h, sq, dh]!
        ctx = q[..., ax.sq.attend(keys=k, values=v, dim=ax.dh, is_causal=True), ax.dh]

        assert_axes(self, ctx, ["b", "h", "sq", "dh"], [2, 4, 8, 16])
        self.assertEqual(ctx.data.shape, (2, 4, 8, 16))

    # -------------------------------------------------------------------------
    # Phase 1: Pointwise Math & Activations
    # -------------------------------------------------------------------------

    def test_pointwise_math(self):
        x = tensor(jnp.array([1.0, 4.0, 9.0], dtype=jnp.float32), ax.d)

        cases = [
            ("exp", jnp.exp(x.data)),
            ("log", jnp.log(x.data)),
            ("abs", jnp.abs(x.data)),
            ("rsqrt", jax.lax.rsqrt(x.data)),
        ]

        for op_name, expected in cases:
            with self.subTest(op_name=op_name):
                token = getattr(ax.d, op_name)()
                y = x[token]
                assert_axes(self, y, ["d"], [3])
                assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_clamp_op(self):
        x = tensor(jnp.array([-5.0, 0.0, 5.0], dtype=jnp.float32), ax.d)
        y = x[ax.d.clamp(-1.0, 1.0)]

        expected = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        assert_axes(self, y, ["d"], [3])
        assert_allclose(y.data, expected)

    def test_swiglu_op_halves_dimension(self):
        x = tensor(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), ax.d)
        y = x[ax.d.swiglu()]

        chunk1, chunk2 = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
        expected = jax.nn.silu(chunk1) * chunk2

        # The axis should automatically be halved!
        assert_axes(self, y, ["d"], [2])
        assert_allclose(y.data, expected)

    # -------------------------------------------------------------------------
    # Phase 2: Advanced Reductions
    # -------------------------------------------------------------------------

    def test_logsumexp_reduction(self):
        x = tensor(jnp.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=jnp.float32), ax.b, ax.d)
        y = x[ax.b, ax.d.logsumexp()]

        expected = jsp.special.logsumexp(x.data, axis=1)
        assert_axes(self, y, ["b"], [2])
        assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_argmax_argmin_reductions(self):
        x = tensor(jnp.array([[1.0, 4.0, 2.0], [7.0, 5.0, 9.0]], dtype=jnp.float32), ax.b, ax.d)

        y_max = x[ax.b, ax.d.argmax()]
        y_min = x[ax.b, ax.d.argmin()]

        assert_axes(self, y_max, ["b"], [2])
        assert_axes(self, y_min, ["b"], [2])
        assert_allclose(y_max.data, jnp.argmax(x.data, axis=1))
        assert_allclose(y_min.data, jnp.argmin(x.data, axis=1))

    def test_any_all_reductions(self):
        x = tensor(jnp.array([[True, False], [True, True]]), ax.b, ax.d)

        y_any = x[ax.b, ax.d.any()]
        y_all = x[ax.b, ax.d.all()]

        assert_axes(self, y_any, ["b"], [2])
        assert_axes(self, y_all, ["b"], [2])

        self.assertTrue(jnp.array_equal(y_any.data, jnp.array([True, True])))
        self.assertTrue(jnp.array_equal(y_all.data, jnp.array([False, True])))

    # -------------------------------------------------------------------------
    # Phase 3: Sparse Routing, Control, & Structural
    # -------------------------------------------------------------------------

    def test_stop_gradient_is_noop_on_forward_pass(self):
        x = tensor(jnp.array([1.0, 2.0], dtype=jnp.float32), ax.d)
        y = x[ax.d.stop_gradient()]

        assert_axes(self, y, ["d"], [2])
        assert_allclose(y.data, x.data)

    def test_scatter_op_update_and_add(self):
        x = tensor(jnp.zeros(4, dtype=jnp.float32), ax.d)
        idx = tensor(jnp.array([1, 3], dtype=jnp.int32), ax.s)
        updates = tensor(jnp.array([10.0, 20.0], dtype=jnp.float32), ax.s)

        # Test mode="update"
        y_update = x[ax.d.scatter(idx, updates, mode="update")]
        expected_update = jnp.array([0.0, 10.0, 0.0, 20.0], dtype=jnp.float32)
        assert_axes(self, y_update, ["d"], [4])
        assert_allclose(y_update.data, expected_update)

        # Test mode="add"
        x_base = tensor(jnp.ones(4, dtype=jnp.float32), ax.d)
        y_add = x_base[ax.d.scatter(idx, updates, mode="add")]
        expected_add = jnp.array([1.0, 11.0, 1.0, 21.0], dtype=jnp.float32)
        assert_allclose(y_add.data, expected_add)

    def test_structural_split(self):
        x = tensor(jnp.arange(12).reshape(2, 6), ax.b, ax.s)

        # Split sequence into 2 chunks of size 3
        chunks = x.split(ax.s, 2)

        self.assertEqual(len(chunks), 2)
        assert_axes(self, chunks[0], ["b", "s"], [2, 3])
        assert_axes(self, chunks[1], ["b", "s"], [2, 3])

        expected_chunk_0 = jnp.array([[0, 1, 2], [6, 7, 8]])
        expected_chunk_1 = jnp.array([[3, 4, 5], [9, 10, 11]])

        assert_allclose(chunks[0].data, expected_chunk_0)
        assert_allclose(chunks[1].data, expected_chunk_1)

    def test_structural_unbind(self):
        # Simulate a fused QKV projection: [batch=2, qkv=3, seq=4, dim=5]
        data = jnp.arange(120, dtype=jnp.float32).reshape(2, 3, 4, 5)
        x = tensor(data, ax.b, ax.qkv(3), ax.s, ax.d)

        # Unbind along the qkv axis!
        q, k, v = x.unbind(ax.qkv)

        # 1. The qkv axis should be completely stripped from the logical layout
        assert_axes(self, q, ["b", "s", "d"], [2, 4, 5])
        assert_axes(self, k, ["b", "s", "d"], [2, 4, 5])
        assert_axes(self, v, ["b", "s", "d"], [2, 4, 5])

        # 2. The data should perfectly match the physical slices
        assert_allclose(q.data, data[:, 0, :, :])
        assert_allclose(k.data, data[:, 1, :, :])
        assert_allclose(v.data, data[:, 2, :, :])

    def test_unbind_axis_not_found_raises(self):
        x = tensor(jnp.zeros((2, 4), dtype=jnp.float32), ax.b, ax.d)

        with self.assertRaises(AxiomShapeError):
            _ = x.unbind(ax.qkv)

    def test_structural_concat(self):
        from axiom import AxiomTensor

        t1 = tensor(jnp.ones((2, 3)), ax.b, ax.s)
        t2 = tensor(jnp.zeros((2, 4)), ax.b, ax.s)

        y = AxiomTensor.concat([t1, t2], axis=ax.s)

        assert_axes(self, y, ["b", "s"], [2, 7])
        self.assertEqual(y.data.shape, (2, 7))
        assert_allclose(y.data[:, :3], jnp.ones((2, 3)))
        assert_allclose(y.data[:, 3:], jnp.zeros((2, 4)))

    def test_structural_topk(self):
        x = tensor(jnp.array([[10.0, 50.0, 20.0, 40.0]], dtype=jnp.float32), ax.b, ax.vocab)

        values, indices = x.topk(ax.vocab, k=2)

        assert_axes(self, values, ["b", "vocab"], [1, 2])
        assert_axes(self, indices, ["b", "vocab"], [1, 2])

        expected_vals = jnp.array([[50.0, 40.0]], dtype=jnp.float32)
        expected_idx = jnp.array([[1, 3]], dtype=jnp.int32)

        assert_allclose(values.data, expected_vals)
        assert_allclose(indices.data, expected_idx)

    def test_associative_scan_linear_recurrence(self):
        # 1. Setup data
        B, S, D = 2, 5, 3
        np.random.seed(42)

        # Standard normal inputs
        x_data = jnp.array(np.random.randn(B, S, D), dtype=jnp.float32)

        # Gates usually between 0 and 1
        a_data = jnp.array(np.random.uniform(0.1, 0.9, (B, S, D)), dtype=jnp.float32)

        X = tensor(x_data, ax.b, ax.s, ax.d)
        A = tensor(a_data, ax.b, ax.s, ax.d)

        # 2. Define the strictly associative combiner for (x, a)
        def linear_combine(i, j):
            x_i, a_i = i
            x_j, a_j = j
            return a_j * x_i + x_j, a_j * a_i

        # 3. Route through Axiom's associative scan (FIXED: Added ax.b and ax.d)
        Y = X[ax.b, ax.s.assoc_scan(linear_combine, inputs=(A,)), ax.d]

        # 4. Compute the ground truth sequentially to prove the parallel math works
        expected_y = np.zeros_like(x_data)
        carry = np.zeros_like(x_data[:, 0, :])

        for t in range(S):
            carry = a_data[:, t, :] * carry + x_data[:, t, :]
            expected_y[:, t, :] = carry

        # 5. Verify shapes and numerical equivalence
        assert_axes(self, Y, ["b", "s", "d"], [B, S, D])
        assert_allclose(Y.data, expected_y, atol=1e-5, rtol=1e-5)

    def test_explicit_conv_1d_causal(self):
        x = tensor(
            jnp.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=jnp.float32),
            ax.b, ax.s, ax.c
        )

        # kernel_size=3, in_features=1, out_features=1
        # all-ones causal FIR: y_t = x_t + x_{t-1} + x_{t-2}
        w = jnp.ones((3, 1, 1), dtype=jnp.float32)

        y = x[..., ax.c.conv(
            3,
            over=ax.s.causal(),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        expected = jnp.array([[[1.0], [3.0], [6.0], [9.0]]], dtype=jnp.float32)
        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 1])
        assert_allclose(y.data, expected)

    def test_explicit_conv_1d_causal_named_kernel_axis(self):
        x = tensor(
            jnp.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=jnp.float32),
            ax.b, ax.s, ax.c
        )

        w = tensor(
            jnp.ones((3, 1, 1), dtype=jnp.float32),
            ax.k(3), ax.c, ax.cout(1)
        )

        y = x[..., ax.c.conv(
            ax.k(3),
            over=ax.s.causal(),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        expected = jnp.array([[[1.0], [3.0], [6.0], [9.0]]], dtype=jnp.float32)
        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 1])
        assert_allclose(y.data, expected)

    def test_explicit_conv_1d_causal_dilated(self):
        x = tensor(
            jnp.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=jnp.float32),
            ax.b, ax.s, ax.c
        )

        # kernel taps at t, t-2, t-4
        w = jnp.ones((3, 1, 1), dtype=jnp.float32)

        y = x[..., ax.c.conv(
            3,
            over=ax.s.causal().dilate(2),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        expected = jnp.array([[[1.0], [2.0], [4.0], [6.0], [9.0]]], dtype=jnp.float32)
        assert_axes(self, y, ["b", "s", "cout"], [1, 5, 1])
        assert_allclose(y.data, expected)

    def test_explicit_conv_1d_causal_depthwise_groups(self):
        x = tensor(
            jnp.array(
                [[[1.0, 10.0],
                  [2.0, 20.0],
                  [3.0, 30.0],
                  [4.0, 40.0]]],
                dtype=jnp.float32
            ),
            ax.b, ax.s, ax.c
        )

        # groups=2 => each output channel only sees its own input channel
        # kernel shape = (K, in_features_per_group, out_features) = (2, 1, 2)
        w = jnp.ones((2, 1, 2), dtype=jnp.float32)

        y = x[..., ax.c.conv(
            2,
            over=ax.s.causal(),
            out=ax.cout(2),
            groups=2,
            weight=w,
            use_bias=False,
        )]

        expected = jnp.array(
            [[[1.0, 10.0],
              [3.0, 30.0],
              [5.0, 50.0],
              [7.0, 70.0]]],
            dtype=jnp.float32
        )

        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 2])
        assert_allclose(y.data, expected)

    def test_explicit_conv_1d_causal_depthwise_string_groups(self):
        x = tensor(
            jnp.array(
                [[[1.0, 10.0],
                  [2.0, 20.0],
                  [3.0, 30.0],
                  [4.0, 40.0]]],
                dtype=jnp.float32
            ),
            ax.b, ax.s, ax.c
        )

        # "depthwise" should resolve to groups=in_features=2
        w = jnp.ones((2, 1, 2), dtype=jnp.float32)

        y = x[..., ax.c.conv(
            2,
            over=ax.s.causal(),
            out=ax.cout(2),
            groups="depthwise",
            weight=w,
            use_bias=False,
        )]

        expected = jnp.array(
            [[[1.0, 10.0],
              [3.0, 30.0],
              [5.0, 50.0],
              [7.0, 70.0]]],
            dtype=jnp.float32
        )

        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 2])
        assert_allclose(y.data, expected)

    def test_implicit_conv_1d_causal(self):
        x = tensor(
            jnp.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=jnp.float32),
            ax.b, ax.s, ax.c
        )

        mod = ImplicitCausalConv1DMod()
        y = mod(x)

        # all-ones implicit kernel_init, no bias
        expected = jnp.array([[[1.0], [3.0], [6.0], [9.0]]], dtype=jnp.float32)
        assert_axes(self, y, ["b", "s", "cout"], [1, 4, 1])
        assert_allclose(y.data, expected)

    def test_conv_same_and_valid_domain_modifiers(self):
        x = tensor(
            jnp.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=jnp.float32),
            ax.b, ax.s, ax.c
        )
        w = jnp.ones((3, 1, 1), dtype=jnp.float32)

        y_same = x[..., ax.c.conv(
            3,
            over=ax.s.same(),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        y_valid = x[..., ax.c.conv(
            3,
            over=ax.s.valid(),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        self.assertEqual(y_same.data.shape, (1, 4, 1))
        self.assertEqual(y_valid.data.shape, (1, 2, 1))

        expected_valid = jnp.array([[[6.0], [9.0]]], dtype=jnp.float32)

        assert_axes(self, y_same, ["b", "s", "cout"], [1, 4, 1])
        assert_axes(self, y_valid, ["b", "s", "cout"], [1, 2, 1])
        assert_allclose(y_valid.data, expected_valid)

    def test_conv_axiswise_causal_is_allowed_in_nd(self):
        x = tensor(jnp.ones((1, 4, 4, 1), dtype=jnp.float32), ax.b, ax.h, ax.w, ax.c)
        w = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

        y = x[..., ax.c.conv(
            (3, 3),
            over=ax.h.causal() & ax.w.same(),
            out=ax.cout(1),
            weight=w,
            use_bias=False,
        )]

        assert_axes(self, y, ["b", "h", "w", "cout"], [1, 4, 4, 1])

    def test_conv_groups_mismatch_raises(self):
        x = tensor(jnp.ones((1, 4, 3), dtype=jnp.float32), ax.b, ax.s, ax.c)

        with self.assertRaises(AxiomShapeError):
            _ = x[..., ax.c.conv(
                3,
                over=ax.s.causal(),
                out=ax.cout(3),
                groups=2,   # 3 input channels not divisible by 2
                weight=jnp.ones((3, 1, 3), dtype=jnp.float32),
                use_bias=False,
            )]

    def test_conv_domain_modifiers_outside_conv_raise(self):
        x = tensor(jnp.ones((1, 4, 1), dtype=jnp.float32), ax.b, ax.s, ax.c)

        with self.assertRaises(AxiomSyntaxError):
            _ = x[ax.b, ax.s.causal(), ax.c]

        with self.assertRaises(AxiomSyntaxError):
            _ = x[ax.b, ax.s.stride(2), ax.c]

        with self.assertRaises(AxiomSyntaxError):
            _ = x[ax.b, ax.s.dilate(2), ax.c]

    def test_pointwise_math_trig_and_square(self):
        # Testing with values where sin and cos have well-known behaviors
        x = tensor(jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi], dtype=jnp.float32), ax.d)

        cases = [
            ("square", jnp.square(x.data)),
            ("sin", jnp.sin(x.data)),
            ("cos", jnp.cos(x.data)),
        ]

        for op_name, expected in cases:
            with self.subTest(op_name=op_name):
                token = getattr(ax.d, op_name)()
                y = x[token]
                assert_axes(self, y, ["d"], [4])
                assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_pointwise_math_rounding(self):
        # Using positive and negative decimals to ensure floor/ceil/round behave correctly
        x = tensor(jnp.array([-1.7, -1.2, 0.0, 1.2, 1.5, 1.7], dtype=jnp.float32), ax.d)

        cases = [
            ("round", jnp.round(x.data)),
            ("floor", jnp.floor(x.data)),
            ("ceil", jnp.ceil(x.data)),
        ]

        for op_name, expected in cases:
            with self.subTest(op_name=op_name):
                token = getattr(ax.d, op_name)()
                y = x[token]
                assert_axes(self, y, ["d"], [6])
                assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    def test_pointwise_math_pow(self):
        x = tensor(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), ax.d)

        # Pow takes an argument, so it is tested separately from the 0-arity ops
        y = x[ax.d.pow(3)]

        expected = jnp.power(x.data, 3)
        assert_axes(self, y, ["d"], [4])
        assert_allclose(y.data, expected, atol=1e-5, rtol=1e-5)

    # -------------------------------------------------------------------------
    # Phase 4: Spatial Pooling
    # -------------------------------------------------------------------------

    def test_pool_1d_max_valid(self):
        # Input: [0, 1, 2, 3, 4, 5]
        x = tensor(jnp.arange(6, dtype=jnp.float32), ax.s)

        # Window 2, stride 2, valid padding
        y = x[ax.s.max_pool(window=2, strides=2, pad="valid")]

        # Expected maxes of [0, 1], [2, 3], [4, 5]
        expected = jnp.array([1.0, 3.0, 5.0], dtype=jnp.float32)

        assert_axes(self, y, ["s"], [3])
        assert_allclose(y.data, expected)

    def test_pool_2d_avg_valid(self):
        # 4x4 matrix
        x = tensor(jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1), ax.b, ax.h, ax.w, ax.c)

        y = x[..., ax.h.avg_pool(2, strides=2), ax.w.avg_pool(2, strides=2), ax.c]

        # 4x4 -> 2x2 average pool
        expected = jnp.array([
            [[[2.5], [4.5]],
             [[10.5], [12.5]]]
        ], dtype=jnp.float32)

        assert_axes(self, y, ["b", "h", "w", "c"], [1, 2, 2, 1])
        assert_allclose(y.data, expected)

    def test_packed_axis_pool_sugar(self):
        # 4x4 matrix
        x = tensor(jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1), ax.b, ax.h, ax.w, ax.c)

        # Notice the * operator! This unpacks the PackedAxis into two separate
        # tokens, keeping the strict 1:1 physical axis constraint perfectly intact!
        y = x[ax.b, *(ax.h & ax.w).max_pool(2, strides=2), ax.c]

        expected = jnp.array([
            [[[5.0], [7.0]],
             [[13.0], [15.0]]]
        ], dtype=jnp.float32)

        assert_axes(self, y, ["b", "h", "w", "c"], [1, 2, 2, 1])
        assert_allclose(y.data, expected)

    def test_inline_assertion_passes_silently(self):
        x = tensor(jnp.zeros((2, 256), dtype=jnp.float32), ax.b, ax.d)

        # Validates physical size 256, does nothing, routes normally
        y = x[ax.b, ax.d == 256]

        assert_axes(self, y, ["b", "d"], [2, 256])
        assert_allclose(y.data, x.data)

    def test_inline_assertion_fails_loudly(self):
        x = tensor(jnp.zeros((2, 128), dtype=jnp.float32), ax.b, ax.d)

        # Physical size is 128, assertion wants 256
        with self.assertRaises(AxiomShapeError) as context:
            _ = x[ax.b, ax.d == 256]

        self.assertIn("Assertion failed", str(context.exception))
        self.assertIn("128", str(context.exception))

    def test_relative_axis_assertion(self):
        # A square matrix (seq_len == dim)
        x = tensor(jnp.zeros((2, 64, 64), dtype=jnp.float32), ax.b, ax.seq, ax.d)

        # Assert that seq is exactly equal to d
        y = x[ax.b, ax.seq == ax.d, ax.d]
        assert_axes(self, y, ["b", "seq", "d"], [2, 64, 64])

        # A rectangular matrix (seq_len != dim)
        x_bad = tensor(jnp.zeros((2, 128, 64), dtype=jnp.float32), ax.b, ax.seq, ax.d)

        with self.assertRaises(AxiomShapeError):
            _ = x_bad[ax.b, ax.seq == ax.d, ax.d]

    def test_packed_axis_assertion(self):
        # We start with a flattened spatial dimension of size 256
        x = tensor(jnp.zeros((2, 256, 3), dtype=jnp.float32), ax.b, ax.flat, ax.c)

        # Assert the flat size is exactly 256, instantly unpack it to h & w, then route
        y = x[ax.b, (ax.flat >> (ax.h(16) & ax.w(16))) == 256, ax.c, "->", ax.b, ax.h, ax.w, ax.c]

        assert_axes(self, y, ["b", "h", "w", "c"], [2, 16, 16, 3])

    def test_bool_trap_prevents_chained_inequalities(self):
        with self.assertRaises(AxiomSyntaxError):
            _ = 0 < ax.d < 512

    # -------------------------------------------------------------------------
    # Constructors & Initialization
    # -------------------------------------------------------------------------

    def test_constructors(self):
        from axiom import AxiomTensor

        # 1. Zeros
        z = AxiomTensor.zeros(ax.b(2), ax.d(4))
        assert_axes(self, z, ["b", "d"], [2, 4])
        assert_allclose(z.data, jnp.zeros((2, 4)))

        # 2. Ones
        o = AxiomTensor.ones(ax.seq(3), ax.d(5), dtype=jnp.float16)
        assert_axes(self, o, ["seq", "d"], [3, 5])
        assert_allclose(o.data, jnp.ones((3, 5)))
        self.assertEqual(o.dtype, jnp.float16)

        # 3. Full
        f = AxiomTensor.full(-3.14, ax.h(2), ax.w(2))
        assert_axes(self, f, ["h", "w"], [2, 2])
        assert_allclose(f.data, jnp.full((2, 2), -3.14))

    def test_constructor_guardrails(self):
        from axiom import AxiomTensor

        # Fails: Missing size
        with self.assertRaises(AxiomShapeError):
            _ = AxiomTensor.zeros(ax.b, ax.d(256))

        # Fails: Symbolic size (can't allocate memory on a promise)
        with self.assertRaises(AxiomShapeError):
            _ = AxiomTensor.ones(ax.b(32), ax.d * 2)

        # Fails: Packed Axis
        with self.assertRaises(TypeError):
            _ = AxiomTensor.zeros(ax.b(32), ax.h(16) & ax.w(16))

    def test_like_constructors(self):
        from axiom import AxiomTensor

        base = AxiomTensor.full(5.0, ax.b(2), ax.d(3))

        z = base.zeros_like()
        assert_axes(self, z, ["b", "d"], [2, 3])
        assert_allclose(z.data, jnp.zeros((2, 3)))

        o = base.ones_like(dtype=jnp.int32)
        assert_axes(self, o, ["b", "d"], [2, 3])
        assert_allclose(o.data, jnp.ones((2, 3)))
        self.assertEqual(o.dtype, jnp.int32)

        f = base.full_like(9.9)
        assert_axes(self, f, ["b", "d"], [2, 3])
        assert_allclose(f.data, jnp.full((2, 3), 9.9))

    # -------------------------------------------------------------------------
    # UnfoldOp & Explicit Convolution Tests
    # -------------------------------------------------------------------------

    def test_basic_1d_unfold(self):
        # Shape: [batch=2, seq=10, channels=1]
        # Data: 0 to 9 so we can visually track the windows
        data = jnp.arange(10, dtype=jnp.float32).reshape(1, 10, 1).repeat(2, axis=0)
        x = tensor(data, ax.b, ax.seq, ax.c)

        # Unfold with window=3, stride=1
        # Expected output sequence length = (10 - 3) // 1 + 1 = 8
        y = x[ax.b, ax.seq.unfold(window=3, out=ax.seq_out & ax.win), ax.c]

        assert_axes(self, y, ["b", "seq_out", "win", "c"], [2, 8, 3, 1])

        # Verify the actual overlapping patches were extracted correctly
        # The first window of the first batch should be [0, 1, 2]
        self.assertEqual(y.data[0, 0, 0, 0], 0.0)
        self.assertEqual(y.data[0, 0, 1, 0], 1.0)
        self.assertEqual(y.data[0, 0, 2, 0], 2.0)

        # The second window should slide by 1: [1, 2, 3]
        self.assertEqual(y.data[0, 1, 0, 0], 1.0)
        self.assertEqual(y.data[0, 1, 2, 0], 3.0)

    def test_2d_im2col_unfold(self):
        # Shape: [batch=2, height=16, width=16, channels=3]
        x = tensor(jnp.zeros((2, 16, 16, 3), dtype=jnp.float32), ax.b, ax.h, ax.w, ax.c)

        # Unfold both spatial dimensions with 3x3 windows
        # Expected spatial out = (16 - 3) // 1 + 1 = 14
        y = x[
            ax.b,
            ax.h.unfold(window=3, out=ax.h_out & ax.wh),
            ax.w.unfold(window=3, out=ax.w_out & ax.ww),
            ax.c
        ]

        # The rank explodes to 6 precisely as requested!
        assert_axes(self, y, ["b", "h_out", "wh", "w_out", "ww", "c"], [2, 14, 3, 14, 3, 3])

    def test_unfold_with_strides_and_dilation(self):
        x = tensor(jnp.zeros((2, 20, 1), dtype=jnp.float32), ax.b, ax.seq, ax.c)

        # Stride=2: (20 - 3) // 2 + 1 = 9
        y_stride = x[ax.b, ax.seq.unfold(window=3, stride=2, out=ax.seq_out & ax.win), ax.c]
        assert_axes(self, y_stride, ["b", "seq_out", "win", "c"], [2, 9, 3, 1])

        # Dilation=2: effective window = (3 - 1) * 2 + 1 = 5
        # Out seq = (20 - 5) // 1 + 1 = 16
        y_dilate = x[ax.b, ax.seq.unfold(window=3, dilation=2, out=ax.seq_out & ax.win), ax.c]
        assert_axes(self, y_dilate, ["b", "seq_out", "win", "c"], [2, 16, 3, 1])

    def test_pad_then_unfold_equivalence(self):
        # Demonstrating how to achieve "same" padding manually
        x = tensor(jnp.zeros((2, 16, 4), dtype=jnp.float32), ax.b, ax.seq, ax.c)

        # 1. Pad 1 on the left, 1 on the right. Sequence becomes 18.
        # 2. Unfold with window 3. Out seq = (18 - 3) // 1 + 1 = 16.
        y = x[ax.b, ax.seq.pad((1, 1)).unfold(window=3, out=ax.seq_out & ax.win), ax.c]

        assert_axes(self, y, ["b", "seq_out", "win", "c"], [2, 16, 3, 4])

    def test_full_explicit_convolution(self):
        from axiom import Module

        # 1. Define the operation inside a Module so Axiom can track the implicit weights!
        class ExplicitConvLayer(Module):
            def __call__(self, x):
                return x[
                    ax.b, ax.seq.pad((1, 1)).unfold(window=3, out=ax.seq_out & ax.win), ax.c,
                    "->",
                    ax.b, ax.seq_out, (ax.win & ax.c).proj(out=ax.cout(8))
                ]

        x = tensor(jnp.zeros((2, 16, 4), dtype=jnp.float32), ax.b, ax.seq, ax.c)

        # 2. Instantiate and run
        layer = ExplicitConvLayer()
        y = layer(x)

        assert_axes(self, y, ["b", "seq_out", "cout"], [2, 16, 8])

    def test_unfold_guardrails(self):
        x = tensor(jnp.zeros((2, 10, 1), dtype=jnp.float32), ax.b, ax.seq, ax.c)

        # Fails: out_axes must be a PackedAxis of exactly two axes
        with self.assertRaises(ValueError):
            _ = x[ax.b, ax.seq.unfold(window=3, out=ax.seq_out), ax.c]

        with self.assertRaises(ValueError):
            _ = x[ax.b, ax.seq.unfold(window=3, out=ax.seq_out & ax.win & ax.extra), ax.c]

        # Fails: Window is larger than the tensor sequence (12 > 10)
        with self.assertRaises(AxiomShapeError) as context:
            _ = x[ax.b, ax.seq.unfold(window=12, out=ax.seq_out & ax.win), ax.c]

        self.assertIn("Tensor is too small", str(context.exception))

    # -------------------------------------------------------------------------
    # Distributed & Vectorization Tests (Phase 4)
    # -------------------------------------------------------------------------

    def test_semantic_vmap_stateless(self):
        from axiom import tensor, vmap

        # 1. A pure function that only knows about Sequence and Channel
        def process_token(x):
            # Layout coming in is [seq, c].
            # We pool the sequence on the LHS, then route to [c, seq] on the RHS!
            return x[ax.seq.max_pool(window=2), ax.c, "->", ax.c, ax.seq]

        # 2. Input data has Batch! Layout: [Batch=4, Seq=10, Channel=8]
        data = jnp.arange(320, dtype=jnp.float32).reshape(4, 10, 8)
        x_batched = tensor(data, ax.b, ax.seq, ax.c)

        # 3. Vectorize over batch
        batched_process = vmap(process_token, over=ax.b)

        # 4. Execute
        y = batched_process(x_batched)

        # 5. Output should be [b, c, seq] where seq is now 5
        assert_axes(self, y, ["b", "c", "seq"], [4, 8, 5])

    def test_semantic_sharding(self):
        from axiom import tensor
        import jax
        import numpy as np

        # Create a mock 1-device supercomputer mesh for the test!
        devices = np.array(jax.devices()).reshape(1, 1)
        mesh = jax.sharding.Mesh(devices, ("data_mesh", "model_mesh"))

        with mesh:
            # 1. Tag axes with hardware meshes
            ax_b_sharded = ax.b.shard("data_mesh")
            ax_d_sharded = ax.d.shard("model_mesh")

            # 2. Create the tensor. Layout is [b, seq, d]
            x = tensor(jnp.zeros((4, 10, 8)), ax_b_sharded, ax.seq, ax_d_sharded)

            # 3. Permute using the routing arrow and apply the sharding constraint!
            y = x[..., "->", ax.seq, ax_d_sharded, ax_b_sharded].apply_sharding()

            # We ensure the tensor survived the JAX constraint barrier
            assert_axes(self, y, ["seq", "d", "b"], [10, 8, 4])

    def test_semantic_slicing_autoregressive(self):
        # Shape (2, 5)
        data = jnp.array([
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51]
        ], dtype=jnp.float32)

        x = tensor(data, ax.b, ax.sq)

        # Shift operations
        inputs = x[..., ax.sq[:-1]]
        targets = x[..., ax.sq[1:]]

        # Verify axes updated correctly
        assert_axes(self, inputs, ["b", "sq"], [2, 4])
        assert_axes(self, targets, ["b", "sq"], [2, 4])

        # Verify data sliced correctly on the physical level
        expected_inputs = jnp.array([[10, 20, 30, 40], [11, 21, 31, 41]], dtype=jnp.float32)
        expected_targets = jnp.array([[20, 30, 40, 50], [21, 31, 41, 51]], dtype=jnp.float32)

        assert_allclose(inputs.data, expected_inputs)
        assert_allclose(targets.data, expected_targets)

    def test_semantic_slicing_rejects_multiple_ellipses(self):
        x = tensor(jnp.zeros((2, 4, 8)), ax.b, ax.sq, ax.d)

        with self.assertRaises(AxiomSyntaxError):
            # Should block standard multiple ellipses
            _ = x[..., ax.sq[1:3], ...]

    def test_apply_sharding_guardrail_raises_without_mesh(self):
        # A tensor explicitly tagged for sharding on "data"
        x = tensor(jnp.zeros((4, 8)), ax.b(4).shard("data"), ax.d(8))

        with self.assertRaisesRegex(RuntimeError, "no JAX Mesh is active"):
            # Should intercept the JAX crash and raise our custom Axiom error
            _ = x.apply_sharding()

    def test_apply_sharding_succeeds_with_mesh(self):
        from jax.sharding import Mesh
        from jax.experimental import mesh_utils

        # Create a mock 1D mesh using local CPU/GPU devices
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices, axis_names=('data',))

        x = tensor(jnp.zeros((4, 8)), ax.b(4).shard("data"), ax.d(8))

        # Inside the context, sharding should apply flawlessly
        with mesh:
            y = x.apply_sharding()
            self.assertEqual(y.data.shape, (4, 8))

    def test_init_linspace_concretization_tracers(self):
        from axiom import init as a_init

        # Mocking an XLA Abstract Tracer that provides a `.val`
        class MockTracer:
            def __init__(self, val):
                self.val = val

        # Simulating JAX passing a tracer shape during jax.eval_shape or remat
        tracer_shape = (MockTracer(5),)

        # Initialize
        init_fn = a_init.linspace(0.0, 1.0)
        rng = jax.random.PRNGKey(0)

        # Should NOT raise a ConcretizationTypeError
        arr = init_fn(rng, tracer_shape)

        self.assertEqual(arr.shape, (5,))
        expected = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32)
        assert_allclose(arr, expected)

    def test_monadic_targeted_mutation(self):
        # Shape: (1, 4)
        data = jnp.array([[10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # Monad action:
        # 1. Open scope on the second half: ax.d[2:]
        # 2. Mutate inner: .square()
        # 3. Close scope: [:]
        y = x[..., ax.d[2:].square()[:]]

        # Expected: First half untouched, second half squared
        assert_axes(self, y, ["b", "d"], [1, 4])
        expected = jnp.array([[10.0, 20.0, 900.0, 1600.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_monadic_scope_escape_and_reduce(self):
        # Shape: (1, 4)
        data = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # Monad action:
        # 1. Open scope on first half: ax.d[:2]
        # 2. Mutate inner: .square()
        # 3. Close scope: [:]
        # 4. Mutate outer (global): .sum()
        y = x[..., ax.d[:2].square()[:].sum()]

        # Expected workflow:
        # inner square: [1.0, 4.0, 3.0, 4.0]
        # outer sum over d: 1 + 4 + 3 + 4 = 12
        assert_axes(self, y, ["b"], [1])
        expected = jnp.array([12.0], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_monadic_scope_escape_and_pad(self):
        # Shape: (1, 3)
        data = jnp.array([[10.0, 20.0, 30.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # Monad action:
        # 1. Open scope on middle element: ax.d[1:2]
        # 2. Mutate inner: .exp()
        # 3. Close scope: [:]
        # 4. Mutate outer (global): .pad()
        y = x[..., ax.d[1:2].exp()[:].pad((1, 1))]

        # Expected workflow:
        # inner exp: [10.0, 400.0 (approx of exp), 30.0]
        # outer pad: [0.0, 10.0, exp(20), 30.0, 0.0]
        assert_axes(self, y, ["b", "d"], [1, 5])

        expected_exp = jnp.exp(20.0)
        expected = jnp.array([[0.0, 10.0, expected_exp, 30.0, 0.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_monadic_dynamic_boundary_and_step(self):
        # Shape: (1, 6)
        data = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # 1. Use a dynamic boundary (ax.d // 2) -> slice ends at index 3
        # 2. Use a step size (::2) -> targets indices 0 and 2
        # 3. Square them and patch them back in
        y = x[..., ax.d[:ax.d // 2:2].square()[:]]

        assert_axes(self, y, ["b", "d"], [1, 6])
        # Indices 0 and 2 (values 1.0 and 3.0) are squared
        expected = jnp.array([[1.0, 2.0, 9.0, 4.0, 5.0, 6.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_monadic_extract_mutated_slice(self):
        # Shape: (1, 4)
        data = jnp.array([[10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # We DO NOT use [:] here. We expect the compiler to return the
        # mutated slice, NOT the patched parent.
        y = x[..., ax.d[2:].square()]

        # The output should just be the squared second half!
        assert_axes(self, y, ["b", "d"], [1, 2])
        expected = jnp.array([[900.0, 1600.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_monadic_scale_explicit(self):
        # Shape: (1, 4)
        data = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # Monad action:
        # 1. Open scope on the middle two elements: ax.d[1:3]
        # 2. Mutate inner: .scale(value=10.0)
        # 3. Close scope: [:]
        y = x[..., ax.d[1:3].scale(value=10.0)[:]]

        # Expected workflow:
        # inner scale: [2.0, 3.0] * 10.0 = [20.0, 30.0]
        # outer patch: [1.0, 20.0, 30.0, 4.0]
        assert_axes(self, y, ["b", "d"], [1, 4])
        expected = jnp.array([[1.0, 20.0, 30.0, 4.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

    def test_full_axis_scale(self):
        # Shape: (1, 3)
        data = jnp.array([[5.0, 10.0, 15.0]], dtype=jnp.float32)
        x = tensor(data, ax.b, ax.d)

        # Standard action: scale the entire axis
        y = x[..., ax.d.scale(value=-2.0)]

        # Expected: Everything multiplied by -2.0
        assert_axes(self, y, ["b", "d"], [1, 3])
        expected = jnp.array([[-10.0, -20.0, -30.0]], dtype=jnp.float32)
        assert_allclose(y.data, expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)