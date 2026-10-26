"""Tests for zero_jax.numpy (jnp) API parity."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from tests.conftest import assert_allclose
from zero_jax import numpy as jnp
from zero_jax._types import ArrayLikeSpec


def test_unary_ops_eager() -> None:
    """Execute ``test_unary_ops_eager``."""
    assert_allclose(jnp.sin(0.0), 0.0)
    assert_allclose(jnp.cos(0.0), 1.0)
    assert_allclose(jnp.exp(0.0), 1.0)
    assert_allclose(jnp.log(1.0), 0.0)


def test_transpose_eager() -> None:
    """Execute ``test_transpose_eager``."""
    x = jnp.array([[1, 2], [3, 4]])
    assert_allclose(jnp.transpose(x), [[1, 3], [2, 4]])


def test_reshape_eager() -> None:
    """Execute ``test_reshape_eager``."""
    x = jnp.array([1, 2, 3, 4])
    assert_allclose(jnp.reshape(x, (2, 2)), [[1, 2], [3, 4]])


def test_broadcast_to_eager() -> None:
    """Execute ``test_broadcast_to_eager``."""
    x = jnp.array([1, 2])
    assert_allclose(jnp.broadcast_to(x, (2, 2)), [[1, 2], [1, 2]])


def test_concatenate_eager() -> None:
    """Execute ``test_concatenate_eager``."""
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    assert_allclose(jnp.concatenate([x, y]), [1, 2, 3, 4])


def test_where_eager() -> None:
    """Execute ``test_where_eager``."""
    cond = jnp.array([True, False])
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    assert_allclose(jnp.where(cond, x, y), [1, 4])


def test_einsum_eager() -> None:
    """Execute ``test_einsum_eager``."""
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    res = jnp.einsum("ij,jk->ik", x, y)
    assert_allclose(res, [[19, 22], [43, 50]])


def test_add_mul_eager() -> None:
    """Execute ``test_add_mul_eager``."""
    assert jnp.add(1, 2) == 3
    assert jnp.multiply(2, 3) == 6


def test_unary_ops_tracing() -> None:
    """Execute ``test_unary_ops_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))

    y = jnp.sin(x)
    assert y.shape == (2,)

    jnp.sin(1.0)

    _tracer.stop_tracing()


def test_transpose_tracing() -> None:
    """Execute ``test_transpose_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 3))
    y = jnp.transpose(x)
    assert y.shape == (3, 2)

    z = jnp.transpose(x, axes=[1, 0])
    assert z.shape == (3, 2)

    jnp.transpose(jnp.array([[1, 2], [3, 4]]))
    _tracer.stop_tracing()


def test_reshape_tracing() -> None:
    """Execute ``test_reshape_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 3))
    y = jnp.reshape(x, (6,))
    assert y.shape == (6,)

    jnp.reshape(jnp.array([1, 2]), (2,))
    _tracer.stop_tracing()


def test_broadcast_to_tracing() -> None:
    """Execute ``test_broadcast_to_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = jnp.broadcast_to(x, (2, 2))
    assert y.shape == (2, 2)

    jnp.broadcast_to(1.0, (2, 2))
    _tracer.stop_tracing()


def test_concatenate_tracing() -> None:
    """Execute ``test_concatenate_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(3,))
    z = jnp.concatenate([x, y])
    assert z.shape == (5,)

    jnp.concatenate([x, jnp.array([1.0])])
    _tracer.stop_tracing()


def test_where_tracing() -> None:
    """Execute ``test_where_tracing``."""
    _tracer.start_tracing()
    cond = ProxyTensor(id="c", shape=(2,))
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(2,))

    z = jnp.where(cond, x, y)
    assert z.shape == (2,)

    jnp.where(True, x, 1.0)
    _tracer.stop_tracing()


def test_einsum_tracing() -> None:
    """Execute ``test_einsum_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 2))
    y = ProxyTensor(id="y", shape=(2, 2))

    jnp.einsum("ij,jk->ik", x, y)

    jnp.einsum("ij,jk->ik", x, jnp.array([[1, 2], [3, 4]]))
    _tracer.stop_tracing()


def test_add_mul_tracing() -> None:
    """Execute ``test_add_mul_tracing``."""
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(2,))

    jnp.add(x, y)
    jnp.add(1.0, x)

    jnp.multiply(x, y)
    jnp.multiply(x, 2.0)
    _tracer.stop_tracing()


def test_missing_numpy_methods() -> None:
    """Execute ``test_missing_numpy_methods``."""
    assert jnp.maximum(1, 2) == 2
    assert jnp.max([1, 2]) == 2
    assert jnp.sum([1, 2]) == 3
    assert jnp.array_equal(jnp.zeros_like(jnp.array([1, 2])), jnp.array([0, 0]))
    assert jnp.array_equal(jnp.zeros((2,)), jnp.array([0.0, 0.0]))
    assert jnp.abs(-1) == 1
    assert jnp.mean([1, 3]) == 2.0
    assert jnp.array_equal(jnp.array([1, 2]), jnp.array([1, 2]))
    assert jnp.isfinite(1)
    assert jnp.allclose(1.0, 1.0)
    assert jnp.array_equal([1], [1])
    assert jnp.broadcast_shapes((1,), (1, 2)) == (1, 2)


def test_multiply_proxy_y() -> None:
    """Execute ``test_multiply_proxy_y``."""
    _tracer.start_tracing()
    y = ProxyTensor(id="y", shape=(2,))
    x = 1.0
    jnp.multiply(x, y)
    _tracer.stop_tracing()


def test_lax_numpy_missing_cov() -> None:
    """Execute ``test_lax_numpy_missing_cov``."""
    assert jnp.max([1, 2], where=[True, False], initial=0) == 1
    assert jnp.sum([1, 2], where=[True, False]) == 1


def test_shape() -> None:
    """Execute ``test_shape``."""
    assert jnp.shape([1, 2, 3]) == (3,)


def test_jnp_trig() -> None:
    """Execute ``test_jnp_trig``."""
    vals = [0.0, 0.5, 1.0]
    x = jnp.array(vals)
    assert_allclose(jnp.acos(x), [math.acos(v) for v in vals])
    assert_allclose(jnp.asin(x), [math.asin(v) for v in vals])
    assert_allclose(jnp.atan(x), [math.atan(v) for v in vals])

    vals_h = [1.0, 2.0, 3.0]
    x_h = jnp.array(vals_h)
    assert_allclose(jnp.acosh(x_h), [math.acosh(v) for v in vals_h])

    vals_sh = [0.0, 1.0, 2.0]
    x_sh = jnp.array(vals_sh)
    assert_allclose(jnp.asinh(x_sh), [math.asinh(v) for v in vals_sh])

    vals_th = [-0.5, 0.0, 0.5]
    x_th = jnp.array(vals_th)
    assert_allclose(jnp.atanh(x_th), [math.atanh(v) for v in vals_th])

    x1 = jnp.array([1.0, -1.0])
    x2 = jnp.array([1.0, 1.0])
    assert_allclose(jnp.atan2(x1, x2), [math.atan2(1.0, 1.0), math.atan2(-1.0, 1.0)])


def test_jnp_bitwise_logical_compare() -> None:
    """Execute ``test_jnp_bitwise_logical_compare``."""
    x1, x2 = jnp.array([True, False, True]), jnp.array([True, True, False])
    assert_allclose(jnp.logical_and(x1, x2), [True, False, False])
    assert_allclose(jnp.logical_or(x1, x2), [True, True, True])
    assert_allclose(jnp.logical_xor(x1, x2), [False, True, True])
    assert_allclose(jnp.logical_not(x1), [False, True, False])

    i1, i2 = jnp.array([1, 2, 3]), jnp.array([1, 4, 3])
    assert_allclose(jnp.bitwise_and(i1, i2), [1 & 1, 2 & 4, 3 & 3])
    assert_allclose(jnp.bitwise_or(i1, i2), [1 | 1, 2 | 4, 3 | 3])
    assert_allclose(jnp.bitwise_xor(i1, i2), [1 ^ 1, 2 ^ 4, 3 ^ 3])
    assert_allclose(jnp.bitwise_not(i1), [~1, ~2, ~3])

    assert_allclose(jnp.equal(i1, i2), [True, False, True])
    assert_allclose(jnp.not_equal(i1, i2), [False, True, False])
    assert_allclose(jnp.greater(i1, i2), [False, False, False])
    assert_allclose(jnp.greater_equal(i1, i2), [True, False, True])
    assert_allclose(jnp.less(i1, i2), [False, True, False])
    assert_allclose(jnp.less_equal(i1, i2), [True, True, True])


def test_jnp_math_numeric() -> None:
    """Execute ``test_jnp_math_numeric``."""
    vals_x = [1.5, 2.5]
    vals_y = [2.0, 3.0]
    x = jnp.array(vals_x, dtype=jnp.float32)
    y = jnp.array(vals_y, dtype=jnp.float32)
    i_array = jnp.array([1, 2], dtype=jnp.int32)

    assert_allclose(jnp.cbrt(x), [v ** (1.0 / 3.0) for v in vals_x])
    assert_allclose(jnp.conj(x), vals_x)
    assert_allclose(jnp.copysign(x, y), [math.copysign(a, b) for a, b in zip(vals_x, vals_y)])
    assert jnp.count_nonzero(x) == 2
    assert_allclose(jnp.cross(x, y), -0.5)
    assert_allclose(jnp.deg2rad(x), [math.radians(v) for v in vals_x])
    assert_allclose(jnp.diag(x), [[1.5, 0.0], [0.0, 2.5]])
    assert_allclose(jnp.fix(x), [1.0, 2.0])
    assert_allclose(jnp.float_power(x, y), [a**b for a, b in zip(vals_x, vals_y)])
    assert_allclose(jnp.fmax(x, y), vals_y)
    assert_allclose(jnp.fmin(x, y), vals_x)
    assert_allclose(jnp.fmod(x, y), [math.fmod(a, b) for a, b in zip(vals_x, vals_y)])

    assert_allclose(jnp.gcd(i_array, i_array), [1, 2])
    assert_allclose(jnp.heaviside(x, y), [1.0, 1.0])
    assert_allclose(jnp.hypot(x, y), [math.hypot(a, b) for a, b in zip(vals_x, vals_y)])
    assert_allclose(jnp.imag(x), [0.0, 0.0])
    assert_allclose(jnp.isclose(x, x), [True, True])
    assert_allclose(jnp.isinf(x), [False, False])
    assert_allclose(jnp.lcm(i_array, i_array), [1, 2])
    assert_allclose(jnp.ldexp(x, i_array), [math.ldexp(1.5, 1), math.ldexp(2.5, 2)])
    assert_allclose(jnp.left_shift(i_array, i_array), [1 << 1, 2 << 2])
    assert_allclose(jnp.logaddexp(x, y), [math.log(math.exp(a) + math.exp(b)) for a, b in zip(vals_x, vals_y)])
    assert_allclose(jnp.logaddexp2(x, y), [math.log2(2.0**a + 2.0**b) for a, b in zip(vals_x, vals_y)])
    assert_allclose(jnp.nextafter(x, y), vals_x, atol=1e-5)
    assert_allclose(jnp.rad2deg(x), [math.degrees(v) for v in vals_x])
    assert_allclose(jnp.real(x), vals_x)
    assert_allclose(jnp.reciprocal(x), [1.0 / v for v in vals_x])
    assert_allclose(jnp.right_shift(i_array, i_array), [1 >> 1, 2 >> 2])
    assert_allclose(jnp.roll(x, shift=1), [2.5, 1.5])
    assert_allclose(jnp.round(x), [2.0, 2.0])
    assert_allclose(
        jnp.sinc(x),
        [math.sin(math.pi * v) / (math.pi * v) for v in vals_x],
    )
    assert_allclose(jnp.sort(x), [1.5, 2.5])

    m = jnp.eye(2)
    assert_allclose(jnp.tril(m), [[1.0, 0.0], [0.0, 1.0]])
    assert_allclose(jnp.triu(m), [[1.0, 0.0], [0.0, 1.0]])

    assert_allclose(jnp.select([x > 1], [x]), vals_x)

    for z_un, n_un in zip(jnp.unstack(x), [1.5, 2.5]):
        assert_allclose(z_un, n_un)


def test_jnp_fft_linalg() -> None:
    """Execute ``test_jnp_fft_linalg``."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    m = jnp.array([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)

    _ = jnp.fft.fft(x)
    _ = jnp.fft.rfft(x)

    _ = jnp.linalg.cholesky(m)
    _ = jnp.linalg.det(m)
    _ = jnp.linalg.inv(m)
    _ = jnp.linalg.matrix_power(m, 2)
    _ = jnp.linalg.pinv(m)
    _ = jnp.linalg.qr(m)
    _ = jnp.linalg.slogdet(m)
    _ = jnp.linalg.solve(m, jnp.array([1.0, 2.0], dtype=jnp.float32))
    _ = jnp.linalg.svd(m)


def test_jnp_aliases() -> None:
    """Execute ``test_jnp_aliases``."""
    vals_x = [1.5, -2.5]
    vals_y = [2.0, 3.0]
    x = jnp.array(vals_x, dtype=jnp.float32)
    y = jnp.array(vals_y, dtype=jnp.float32)

    assert_allclose(jnp.absolute(x), [1.5, 2.5])
    assert_allclose(jnp.around(x), [2.0, -2.0])
    assert_allclose(jnp.round_(x), [2.0, -2.0])
    assert_allclose(jnp.conjugate(x), vals_x)
    assert_allclose(jnp.cumulative_sum(x), [1.5, -1.0])
    assert_allclose(jnp.degrees(x), [math.degrees(v) for v in vals_x])
    assert_allclose(jnp.radians(x), [math.radians(v) for v in vals_x])
    assert_allclose(jnp.pow(x, y), [1.5**2.0, (-2.5) ** 3.0])


def test_jnp_more_aliases() -> None:
    """Execute ``test_jnp_more_aliases``."""
    x = jnp.array([1, 2], dtype=jnp.int32)
    y = jnp.array([1, 1], dtype=jnp.int32)
    z = jnp.array([1.0, jnp.nan, -jnp.inf, jnp.inf], dtype=jnp.float32)

    assert_allclose(jnp.bitwise_invert(x), [~1, ~2])
    assert_allclose(jnp.bitwise_left_shift(x, y), [1 << 1, 2 << 1])
    assert_allclose(jnp.bitwise_right_shift(x, y), [1 >> 1, 2 >> 1])
    assert_allclose(jnp.concat([x, y]), [1, 2, 1, 1])
    assert_allclose(jnp.invert(x), [~1, ~2])

    v = jnp.array([1.0, 2.0, 3.0])
    assert_allclose(jnp.nanmax(v), 3.0)
    assert_allclose(jnp.nanmin(v), 1.0)
    assert_allclose(jnp.nanprod(v), 6.0)
    assert_allclose(jnp.nansum(v), 6.0)

    assert_allclose(jnp.isneginf(z), [False, False, True, False])
    assert_allclose(jnp.isposinf(z), [False, False, False, True])


def test_jnp_frexp() -> None:
    """Execute ``test_jnp_frexp``."""
    vals = [1.5, 2.5]
    x = jnp.array(vals, dtype=jnp.float32)
    a, b = jnp.frexp(x)
    expected_pairs = [math.frexp(v) for v in vals]
    assert_allclose(a, [p[0] for p in expected_pairs])
    assert_allclose(b, [p[1] for p in expected_pairs])


def test_jnp_nan_to_num_searchsorted_signbit() -> None:
    """Execute ``test_jnp_nan_to_num_searchsorted_signbit``."""
    x = jnp.array([1.0, jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float32)
    res_nan = jnp.nan_to_num(x)
    assert res_nan.shape == (4,)
    assert_allclose(jnp.signbit(jnp.array([1.0, -1.0])), [False, True])

    a = jnp.array([1, 2, 4, 5])
    v = jnp.array([3, 6])
    assert_allclose(jnp.searchsorted(a, v), [2, 4])


def test_jnp_copy() -> None:
    """Execute ``test_jnp_copy``."""
    x = jnp.array([1, 2, 3])
    assert_allclose(jnp.copy(x), [1, 2, 3])


def test_jnp_added_funcs_eager() -> None:
    """Execute ``test_jnp_added_funcs_eager``."""
    x = jnp.array([1, 2])
    assert_allclose(jnp.append(x, [3, 4]), [1, 2, 3, 4])

    y = jnp.astype(x, jnp.float32)
    assert str(y.dtype) == "float32" or y.dtype == jnp.float32

    res1 = jnp.atleast_1d(1)
    assert res1.ndim >= 1

    res2 = jnp.atleast_2d(1)
    assert res2.ndim >= 2

    res3 = jnp.atleast_3d(1)
    assert res3.ndim >= 3

    assert_allclose(jnp.average([1, 2, 3]), 2.0)

    A = jnp.eye(2) * 2
    B = jnp.eye(3) * 3
    blk = jnp.block([[A, jnp.zeros((2, 3))], [jnp.zeros((3, 2)), B]])
    assert blk.shape == (5, 5)


def test_jnp_batch2_eager() -> None:
    """Execute ``test_jnp_batch2_eager``."""

    def my_func(a: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``my_func``.

        Args:
            a: Parameter ``a``.
        """
        return jnp.sum(a, axis=0)

    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    res = jnp.apply_along_axis(my_func, 0, x)
    assert_allclose(res, [5, 7, 9])

    res3 = jnp.argpartition(jnp.array([3, 4, 2, 1]), 2)
    assert res3.shape == (4,)

    res4 = jnp.argwhere(x > 3)
    assert_allclose(res4, [[1, 0], [1, 1], [1, 2]])

    res5 = jnp.choose(jnp.array([0, 1, 0]), [jnp.array([10, 20, 30]), jnp.array([40, 50, 60])])
    assert_allclose(res5, [10, 50, 30])

    res6 = jnp.column_stack((jnp.array([1, 2]), jnp.array([3, 4])))
    assert_allclose(res6, [[1, 3], [2, 4]])

    res7 = jnp.compress(jnp.array([True, False, True]), jnp.array([1, 2, 3]))
    assert_allclose(res7, [1, 3])

    res8 = jnp.convolve(jnp.array([1.0, 2.0, 3.0]), jnp.array([0.0, 1.0, 0.5]))
    assert_allclose(res8, [0.0, 1.0, 2.5, 4.0, 1.5])

    res9 = jnp.corrcoef(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert_allclose(res9, [[1.0, 1.0], [1.0, 1.0]])

    res10 = jnp.correlate(jnp.array([1.0, 2.0, 3.0]), jnp.array([0.0, 1.0]))
    assert_allclose(res10, [2.0, 3.0])

    res11 = jnp.cov(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert_allclose(res11, [[1.0, 1.0], [1.0, 1.0]])


def test_jnp_batch3_eager() -> None:
    """Execute ``test_jnp_batch3_eager``."""
    x = jnp.array([1, 2, 3])
    y = jnp.array([1, 2, 3])
    assert bool(jnp.array_equiv(x, y)) is True

    assert repr(jnp.array_repr(x)) != ""
    assert repr(jnp.array_str(x)) != ""

    assert_allclose(jnp.bartlett(3), [0.0, 1.0, 0.0])
    assert_allclose(jnp.bincount(jnp.array([1, 1, 2])), [0, 2, 1])
    assert_allclose(jnp.bitwise_count(jnp.array([1, 2, 3])), [1, 1, 2])
    assert_allclose(jnp.blackman(3), [0.0, 1.0, 0.0], atol=1e-6)

    res1, res2 = jnp.broadcast_arrays(jnp.array([1, 2]), jnp.array([[1], [2]]))
    assert_allclose(res1, [[1, 2], [1, 2]])
    assert_allclose(res2, [[1, 1], [2, 2]])

    assert bool(jnp.can_cast(jnp.int32, jnp.int32)) is True
    assert_allclose(jnp.cumprod(x), [1, 2, 6])
    assert_allclose(jnp.delete(x, 1), [1, 3])

    r1, r2 = jnp.diag_indices(2)
    assert_allclose(r1, [0, 1])
    assert_allclose(r2, [0, 1])

    r1, r2 = jnp.diag_indices_from(jnp.eye(2))
    assert_allclose(r1, [0, 1])
    assert_allclose(r2, [0, 1])

    assert_allclose(jnp.diagflat(jnp.array([1, 2])), [[1, 0], [0, 2]])
    assert_allclose(jnp.diagonal(jnp.eye(2)), [1.0, 1.0])
    assert_allclose(jnp.diff(x), [1, 1])
    assert_allclose(jnp.digitize(jnp.array([0.5, 1.5]), jnp.array([0, 1, 2])), [1, 2])


def test_jnp_batch4_eager() -> None:
    """Execute ``test_jnp_batch4_eager``."""
    x = jnp.array([1, 2, 3])
    assert jnp.dtype("int32") is not None
    assert_allclose(jnp.ediff1d(x), [1, 1])

    res1, res2 = jnp.einsum_path("i,i->", x, x)
    assert res1 is not None and res2 is not None

    assert_allclose(jnp.extract(x > 1, x), [2, 3])
    assert_allclose(jnp.fabs(jnp.array([-1.5, 2.5])), [1.5, 2.5])

    res = jnp.fill_diagonal(jnp.zeros((3, 3)), 1, inplace=False)
    assert_allclose(res, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert callable(jnp.finfo)
    assert_allclose(jnp.flatnonzero(jnp.array([-1, 0, 1])), [0, 2])
    assert_allclose(jnp.flip(x), [3, 2, 1])
    assert_allclose(jnp.fliplr(jnp.eye(2)), [[0.0, 1.0], [1.0, 0.0]])
    assert_allclose(jnp.flipud(jnp.eye(2)), [[0.0, 1.0], [1.0, 0.0]])
    assert jnp.float8_e4m3fn == jnp.float16


def test_jnp_batch5_eager() -> None:
    """Execute ``test_jnp_batch5_eager``."""
    b = bytes([1, 2, 3, 4])
    assert_allclose(jnp.frombuffer(b, dtype=jnp.uint8), [1, 2, 3, 4])

    assert_allclose(
        jnp.fromfunction(lambda i, j: i == j, (3, 3), dtype=int),
        [[True, False, False], [False, True, False], [False, False, True]],
    )

    iterable = (val * val for val in range(5))
    assert_allclose(jnp.fromiter(iterable, float), [0.0, 1.0, 4.0, 9.0, 16.0])

    assert_allclose(jnp.fromstring("1 2", dtype=int, sep=" "), [1, 2])
    assert_allclose(jnp.geomspace(1, 1000, num=4), [1.0, 10.0, 100.0, 1000.0])
    assert callable(jnp.get_printoptions)
    assert_allclose(jnp.gradient(jnp.array([1.0, 2.0, 4.0, 7.0, 11.0])), [1.0, 1.5, 2.5, 3.5, 4.0])
    assert_allclose(jnp.hamming(3), [0.08, 1.0, 0.08], atol=1e-2)
    assert_allclose(jnp.hanning(3), [0.0, 1.0, 0.0], atol=1e-6)

    h1, e1 = jnp.histogram(jnp.array([1, 2, 1]), bins=[0, 1, 2, 3])
    assert_allclose(h1, [0, 2, 1])
    assert_allclose(e1, [0, 1, 2, 3])

    h2d, _x1, _y1 = jnp.histogram2d(jnp.array([1, 2, 1]), jnp.array([1, 2, 1]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]])
    assert h2d.shape == (3, 3)

    assert jnp.histogram_bin_edges(jnp.array([1, 2])).shape[0] >= 2

    hdd, _ = jnp.histogramdd(jnp.array([[1, 1], [2, 2], [1, 1]]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]])
    assert hdd.shape == (3, 3)

    assert jnp.i0(jnp.array([0.0, 1.0, 2.0])).shape == (3,)
    assert callable(jnp.iinfo)

    r1, _r2 = jnp.indices((2, 3), sparse=True)
    assert_allclose(r1, [[0], [1]])

    assert_allclose(jnp.insert(jnp.array([1, 2, 3]), 1, 5), [1, 5, 2, 3])
    assert callable(jnp.interp)
    assert callable(jnp.intersect1d)


def test_jnp_batch6_eager() -> None:
    """Execute ``test_jnp_batch6_eager``."""
    x = jnp.array([1, 2, 3])

    assert_allclose(jnp.iscomplex(x), [False, False, False])
    assert bool(jnp.iscomplexobj(x)) is False
    assert bool(jnp.isdtype(jnp.int32, jnp.int32)) is True
    assert_allclose(jnp.isin(x, [1, 3]), [True, False, True])
    assert_allclose(jnp.isreal(x), [True, True, True])
    assert bool(jnp.isscalar(3.1)) is True

    assert callable(jnp.ix_)

    assert jnp.kaiser(3, 14).shape == (3,)
    assert callable(jnp.kron)
    assert callable(jnp.lexsort)

    assert callable(jnp.load) and "load" in dir(jnp)

    m = jnp.eye(2)
    assert callable(jnp.matrix_transpose)
    assert callable(jnp.median)

    mod_f, mod_i = jnp.modf(jnp.array([1.5, 2.5]))
    assert_allclose(mod_f, [0.5, 0.5])
    assert_allclose(mod_i, [1.0, 2.0])

    nx = jnp.array([1.0, jnp.nan, 3.0])
    assert callable(jnp.nanargmax)
    assert callable(jnp.nanargmin)
    assert callable(jnp.nancumprod)
    assert callable(jnp.nancumsum)
    assert_allclose(jnp.nanmean(nx), 2.0)
    assert callable(jnp.nanmedian)
    assert callable(jnp.nanpercentile)
    assert callable(jnp.nanquantile)
    assert callable(jnp.nanstd)
    assert callable(jnp.nanvar)
    assert m.shape == (2, 2)
