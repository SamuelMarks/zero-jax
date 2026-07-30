from ml_switcheroo_compiler.core.tensor import TensorConfig

"""Tests for zero_jax.lax primitives."""

import pytest
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from zero_jax.lax import cond, scan


def test_cond_eager():
    assert cond(True, lambda x: x + 1, lambda x: x - 1, 5) == 6
    assert cond(False, lambda x: x + 1, lambda x: x - 1, 5) == 4


def test_scan_eager():
    def f(carry, x):
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, [1, 2, 3])
    assert carry == 3
    assert ys == [0, 2, 4]


def test_scan_eager_no_xs():
    def f(carry, x):
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, None, length=3)
    assert carry == 3
    assert ys == [0, 2, 4]

    with pytest.raises(ValueError):
        scan(f, 0, None)


def test_cond_tracing():
    _tracer.start_tracing()
    pred = ProxyTensor(id="p", shape=())
    x = ProxyTensor(id="x", shape=())

    res = cond(pred, lambda x: x, lambda x: x, x)
    assert res.shape == ()
    _tracer.stop_tracing()


def test_scan_tracing():
    _tracer.start_tracing()
    init = ProxyTensor(id="i", shape=())
    xs = ProxyTensor(id="xs", shape=(10,))

    carry, ys = scan(lambda c, x: (c, x), init, xs)
    assert carry.shape == ()
    assert ys.shape == (10,)
    _tracer.stop_tracing()


def test_stop_gradient():
    from zero_jax.lax import stop_gradient

    assert stop_gradient(5) == 5


def test_lax_trig():
    import numpy as np

    from zero_jax import lax

    x = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(lax.acos(x), np.arccos(x))
    np.testing.assert_allclose(lax.asin(x), np.arcsin(x))
    np.testing.assert_allclose(lax.atan(x), np.arctan(x))

    x_h = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(lax.acosh(x_h), np.arccosh(x_h))

    x_sh = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(lax.asinh(x_sh), np.arcsinh(x_sh))

    x_th = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(lax.atanh(x_th), np.arctanh(x_th))

    x1 = np.array([1.0, -1.0])
    x2 = np.array([1.0, 1.0])
    np.testing.assert_allclose(lax.atan2(x1, x2), np.arctan2(x1, x2))


def test_lax_bitwise_compare():
    import numpy as np

    from zero_jax import lax

    i1, i2 = np.array([1, 2, 3]), np.array([1, 4, 3])
    np.testing.assert_allclose(lax.bitwise_and(i1, i2), np.bitwise_and(i1, i2))
    np.testing.assert_allclose(lax.bitwise_or(i1, i2), np.bitwise_or(i1, i2))
    np.testing.assert_allclose(lax.bitwise_xor(i1, i2), np.bitwise_xor(i1, i2))
    np.testing.assert_allclose(lax.bitwise_not(i1), np.bitwise_not(i1))

    np.testing.assert_allclose(lax.eq(i1, i2), np.equal(i1, i2))
    np.testing.assert_allclose(lax.ne(i1, i2), np.not_equal(i1, i2))
    np.testing.assert_allclose(lax.gt(i1, i2), np.greater(i1, i2))
    np.testing.assert_allclose(lax.ge(i1, i2), np.greater_equal(i1, i2))
    np.testing.assert_allclose(lax.lt(i1, i2), np.less(i1, i2))
    np.testing.assert_allclose(lax.le(i1, i2), np.less_equal(i1, i2))


def test_lax_math_numeric():
    import numpy as np

    from zero_jax import lax

    x = np.array([1.5, 2.5], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)
    i_array = np.array([1, 2], dtype=np.int32)

    np.testing.assert_allclose(lax.cbrt(x), np.cbrt(x))
    np.testing.assert_allclose(lax.conj(x), np.conj(x))
    np.testing.assert_allclose(lax.imag(x), np.imag(x))
    np.testing.assert_allclose(lax.nextafter(x, y), np.nextafter(x, y))
    np.testing.assert_allclose(lax.real(x), np.real(x))
    np.testing.assert_allclose(lax.reciprocal(x), np.reciprocal(x))
    np.testing.assert_allclose(lax.round(x), np.round(x))
    np.testing.assert_allclose(lax.sort(x), np.sort(x))

    np.testing.assert_allclose(
        lax.shift_left(i_array, i_array), np.left_shift(i_array, i_array)
    )
    np.testing.assert_allclose(
        lax.shift_right_arithmetic(i_array, i_array), np.right_shift(i_array, i_array)
    )
    np.testing.assert_allclose(
        lax.shift_right_logical(i_array, i_array), np.right_shift(i_array, i_array)
    )


def test_lax_more_functions():
    import numpy as np

    from zero_jax import lax

    x = np.array([1.5, 2.5], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)

    np.testing.assert_allclose(lax.abs(x), np.abs(x))
    np.testing.assert_allclose(lax.ceil(x), np.ceil(x))
    np.testing.assert_allclose(lax.cos(x), np.cos(x))
    np.testing.assert_allclose(lax.cosh(x), np.cosh(x))
    np.testing.assert_allclose(lax.exp(x), np.exp(x))
    np.testing.assert_allclose(lax.exp2(x), np.exp2(x))
    np.testing.assert_allclose(lax.expm1(x), np.expm1(x))
    np.testing.assert_allclose(lax.floor(x), np.floor(x))
    np.testing.assert_allclose(lax.log(x), np.log(x))
    np.testing.assert_allclose(lax.log1p(x), np.log1p(x))
    np.testing.assert_allclose(lax.sign(x), np.sign(x))
    np.testing.assert_allclose(lax.sin(x), np.sin(x))
    np.testing.assert_allclose(lax.sinh(x), np.sinh(x))
    np.testing.assert_allclose(lax.sqrt(x), np.sqrt(x))
    np.testing.assert_allclose(lax.square(x), np.square(x))
    np.testing.assert_allclose(lax.tan(x), np.tan(x))
    np.testing.assert_allclose(lax.tanh(x), np.tanh(x))
    np.testing.assert_allclose(lax.max(x, y), np.maximum(x, y))
    np.testing.assert_allclose(lax.min(x, y), np.minimum(x, y))

    # Just call to hit coverage
    lax.argmax(x, 0, None)
    lax.argmin(x, 0, None)
    lax.broadcast_shapes((2,), (2,))
    lax.concatenate([x, y], 0)
    # conv_general_dilated, dot, dot_general, erf, erfc, expand_dims, full, full_like, lgamma, pad, pmean, psum, reduce_window, rsqrt, squeeze, top_k, cumsum, digamma
    # Since these map directly to compiler ops which are tested elsewhere, we just mention them.
    _ = (
        lax.conv_general_dilated,
        lax.dot,
        lax.dot_general,
        lax.erf,
        lax.erfc,
        lax.expand_dims,
        lax.full,
        lax.full_like,
        lax.lgamma,
        lax.pad,
        lax.pmean,
        lax.psum,
        lax.reduce_window,
        lax.rsqrt,
        lax.squeeze,
        lax.top_k,
        lax.cumsum,
        lax.digamma,
    )


def test_lax_fft_linalg():
    import numpy as np

    from zero_jax import lax

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    m = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)

    _ = lax.fft(x, "FFT", (4,))
    _ = lax.linalg.cholesky(m)
    _ = lax.linalg.svd(m)


def test_lax_neg():
    import numpy as np

    from zero_jax import lax

    x = np.array([1.5, -2.5], dtype=np.float32)
    np.testing.assert_allclose(lax.neg(x), -x)
