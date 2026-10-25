"""Tests for zero_jax.lax primitives."""

from __future__ import annotations

import math

import pytest
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from tests.conftest import assert_allclose
from zero_jax import lax
from zero_jax import numpy as jnp
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax import cond, scan


def test_cond_eager() -> None:
    """Execute ``test_cond_eager``."""
    assert cond(True, lambda x: x + 1, lambda x: x - 1, 5) == 6
    assert cond(False, lambda x: x + 1, lambda x: x - 1, 5) == 4


def test_scan_eager() -> None:
    """Execute ``test_scan_eager``."""

    def f(carry: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            carry: Parameter ``carry``.
            x: Parameter ``x``.
        """
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, [1, 2, 3])
    assert carry == 3
    assert ys == [0, 2, 4]


def test_scan_eager_no_xs() -> None:
    """Execute ``test_scan_eager_no_xs``."""

    def f(carry: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            carry: Parameter ``carry``.
            x: Parameter ``x``.
        """
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, None, length=3)
    assert carry == 3
    assert ys == [0, 2, 4]

    with pytest.raises(ValueError):
        scan(f, 0, None)


def test_cond_tracing() -> None:
    """Execute ``test_cond_tracing``."""
    _tracer.start_tracing()
    pred = ProxyTensor(id="p", shape=())
    x = ProxyTensor(id="x", shape=())

    res = cond(pred, lambda x: x, lambda x: x, x)
    assert res.shape == ()
    _tracer.stop_tracing()


def test_scan_tracing() -> None:
    """Execute ``test_scan_tracing``."""
    _tracer.start_tracing()
    init = ProxyTensor(id="i", shape=())
    xs = ProxyTensor(id="xs", shape=(10,))

    carry, ys = scan(lambda c, x: (c, x), init, xs)
    assert carry.shape == ()
    assert ys.shape == (10,)
    _tracer.stop_tracing()


def test_stop_gradient() -> None:
    """Execute ``test_stop_gradient``."""
    from zero_jax.lax import stop_gradient

    assert stop_gradient(5) == 5


def test_lax_trig() -> None:
    """Execute ``test_lax_trig``."""
    vals = [0.0, 0.5, 1.0]
    x = jnp.array(vals)
    assert_allclose(lax.acos(x), [math.acos(v) for v in vals])
    assert_allclose(lax.asin(x), [math.asin(v) for v in vals])
    assert_allclose(lax.atan(x), [math.atan(v) for v in vals])

    vals_h = [1.0, 2.0, 3.0]
    x_h = jnp.array(vals_h)
    assert_allclose(lax.acosh(x_h), [math.acosh(v) for v in vals_h])

    vals_sh = [0.0, 1.0, 2.0]
    x_sh = jnp.array(vals_sh)
    assert_allclose(lax.asinh(x_sh), [math.asinh(v) for v in vals_sh])

    vals_th = [-0.5, 0.0, 0.5]
    x_th = jnp.array(vals_th)
    assert_allclose(lax.atanh(x_th), [math.atanh(v) for v in vals_th])

    x1 = jnp.array([1.0, -1.0])
    x2 = jnp.array([1.0, 1.0])
    assert_allclose(lax.atan2(x1, x2), [math.atan2(1.0, 1.0), math.atan2(-1.0, 1.0)])


def test_lax_bitwise_compare() -> None:
    """Execute ``test_lax_bitwise_compare``."""
    i1, i2 = jnp.array([1, 2, 3]), jnp.array([1, 4, 3])
    assert_allclose(lax.bitwise_and(i1, i2), [1 & 1, 2 & 4, 3 & 3])
    assert_allclose(lax.bitwise_or(i1, i2), [1 | 1, 2 | 4, 3 | 3])
    assert_allclose(lax.bitwise_xor(i1, i2), [1 ^ 1, 2 ^ 4, 3 ^ 3])
    assert_allclose(lax.bitwise_not(i1), [~1, ~2, ~3])

    assert_allclose(lax.eq(i1, i2), [True, False, True])
    assert_allclose(lax.ne(i1, i2), [False, True, False])
    assert_allclose(lax.gt(i1, i2), [False, False, False])
    assert_allclose(lax.ge(i1, i2), [True, False, True])
    assert_allclose(lax.lt(i1, i2), [False, True, False])
    assert_allclose(lax.le(i1, i2), [True, True, True])


def test_lax_math_numeric() -> None:
    """Execute ``test_lax_math_numeric``."""
    vals_x = [1.5, 2.5]
    vals_y = [2.0, 3.0]
    x = jnp.array(vals_x, dtype=jnp.float32)
    y = jnp.array(vals_y, dtype=jnp.float32)
    i_array = jnp.array([1, 2], dtype=jnp.int32)

    assert_allclose(lax.cbrt(x), [v ** (1.0 / 3.0) for v in vals_x])
    assert_allclose(lax.conj(x), vals_x)
    assert_allclose(lax.imag(x), [0.0, 0.0])
    assert_allclose(lax.nextafter(x, y), vals_x, atol=1e-5)
    assert_allclose(lax.real(x), vals_x)
    assert_allclose(lax.reciprocal(x), [1.0 / v for v in vals_x])
    assert_allclose(lax.round(x), [2.0, 2.0])
    assert_allclose(lax.sort(x), [1.5, 2.5])

    assert_allclose(lax.shift_left(i_array, i_array), [1 << 1, 2 << 2])
    assert_allclose(lax.shift_right_arithmetic(i_array, i_array), [1 >> 1, 2 >> 2])
    assert_allclose(lax.shift_right_logical(i_array, i_array), [1 >> 1, 2 >> 2])


def test_lax_more_functions() -> None:
    """Execute ``test_lax_more_functions``."""
    vals_x = [1.5, 2.5]
    vals_y = [2.0, 3.0]
    x = jnp.array(vals_x, dtype=jnp.float32)
    y = jnp.array(vals_y, dtype=jnp.float32)

    assert_allclose(lax.abs(x), [abs(v) for v in vals_x])
    assert_allclose(lax.ceil(x), [math.ceil(v) for v in vals_x])
    assert_allclose(lax.cos(x), [math.cos(v) for v in vals_x])
    assert_allclose(lax.cosh(x), [math.cosh(v) for v in vals_x])
    assert_allclose(lax.exp(x), [math.exp(v) for v in vals_x])
    assert_allclose(lax.exp2(x), [2.0**v for v in vals_x])
    assert_allclose(lax.expm1(x), [math.expm1(v) for v in vals_x])
    assert_allclose(lax.floor(x), [math.floor(v) for v in vals_x])
    assert_allclose(lax.log(x), [math.log(v) for v in vals_x])
    assert_allclose(lax.log1p(x), [math.log1p(v) for v in vals_x])
    assert_allclose(lax.sign(x), [1.0, 1.0])
    assert_allclose(lax.sin(x), [math.sin(v) for v in vals_x])
    assert_allclose(lax.sinh(x), [math.sinh(v) for v in vals_x])
    assert_allclose(lax.sqrt(x), [math.sqrt(v) for v in vals_x])
    assert_allclose(lax.square(x), [v * v for v in vals_x])
    assert_allclose(lax.tan(x), [math.tan(v) for v in vals_x])
    assert_allclose(lax.tanh(x), [math.tanh(v) for v in vals_x])
    assert_allclose(lax.max(x, y), vals_y)
    assert_allclose(lax.min(x, y), vals_x)

    lax.argmax(x, 0, None)
    lax.argmin(x, 0, None)
    lax.broadcast_shapes((2,), (2,))
    lax.concatenate([x, y], 0)
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


def test_lax_fft_linalg() -> None:
    """Execute ``test_lax_fft_linalg``."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    m = jnp.array([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)

    _ = lax.fft(x, "FFT", (4,))
    _ = lax.linalg.cholesky(m)
    _ = lax.linalg.svd(m)


def test_lax_neg() -> None:
    """Execute ``test_lax_neg``."""
    x = jnp.array([1.5, -2.5], dtype=jnp.float32)
    assert_allclose(lax.neg(x), [-1.5, 2.5])
