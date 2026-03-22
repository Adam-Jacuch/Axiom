import unittest

import jax
import jax.numpy as jnp
import numpy as np

from axiom import ax, tensor, Module
from axiom.exceptions import AxiomShapeError, AxiomSyntaxError
from axiom.core.axis import SymbolicSize


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


if __name__ == "__main__":
    unittest.main(verbosity=2)