"""Tests for zero_jax.nn.activation."""

from __future__ import annotations

import math

from tests.conftest import assert_allclose
from zero_jax import nn
from zero_jax import numpy as jnp
from zero_jax.nn.activation import _erf, gelu, log_sigmoid, log_softmax, logsumexp, one_hot, selu, sigmoid, softmax


def test_gelu() -> None:
    """Test gelu."""
    x = jnp.array([-1.0, 0.0, 1.0])
    y_approx = gelu(x, approximate=True)
    y_exact = gelu(x, approximate=False)
    assert y_approx.shape == x.shape
    assert y_exact.shape == x.shape


def test_logsumexp() -> None:
    """Test logsumexp."""
    x = jnp.array([1.0, 2.0, 3.0])
    res = logsumexp(x)
    assert isinstance(res, float) or res.shape is not None

    _res2, sign2 = logsumexp(
        x,
        b=jnp.array([1.0, 1.0, 1.0]),
        keepdims=True,
        return_sign=True,
        where=jnp.array([True, True, False]),
    )
    assert sign2 == 1.0

    x_inf = jnp.array([-jnp.inf, -jnp.inf])
    logsumexp(x_inf)


def test_one_hot() -> None:
    """Test one_hot."""
    x = jnp.array([0, 1, 2])
    y = one_hot(x, num_classes=3)
    assert y.shape == (3, 3)

    y2 = one_hot(x, num_classes=3, axis=0)
    assert y2.shape == (3, 3)


def test_softmax() -> None:
    """Test softmax."""
    x = jnp.array([1.0, 2.0, 3.0])
    y = softmax(x)
    assert_allclose(jnp.sum(y), 1.0)

    y2 = softmax(x, where=jnp.array([True, False, True]))
    assert y2[1] == 0.0


def test_sigmoid() -> None:
    """Execute ``test_sigmoid``."""
    x = jnp.array([0.0])
    assert_allclose(sigmoid(x), [0.5])


def test_log_sigmoid() -> None:
    """Execute ``test_log_sigmoid``."""
    x = jnp.array([0.0])
    res = log_sigmoid(x)
    assert res < 0.0


def test_gelu_eager_non_approx() -> None:
    """Execute ``test_gelu_eager_non_approx``."""
    x = jnp.array([0.0, 1.0, -1.0])
    res2 = gelu(x, approximate=False)
    assert res2.shape == (3,)


def test_logsumexp_eager_extra() -> None:
    """Execute ``test_logsumexp_eager_extra``."""
    x = jnp.array([0.0, 1.0, 2.0])
    _res, sign = logsumexp(x, b=jnp.array([1, 1, 1]), where=jnp.array([True, True, False]), return_sign=True)
    assert sign > 0


def test_softmax_eager_extra() -> None:
    """Execute ``test_softmax_eager_extra``."""
    x = jnp.array([0.0, 1.0, 2.0])
    res2 = softmax(x, where=jnp.array([True, True, False]))
    assert res2.shape == (3,)


def test_selu() -> None:
    """Execute ``test_selu``."""
    x = jnp.array([0.0, 1.0, -1.0])
    res = selu(x)
    assert res.shape == x.shape


def test_log_softmax() -> None:
    """Execute ``test_log_softmax``."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    res = log_softmax(x, axis=-1)
    assert res.shape == x.shape


def test_erf() -> None:
    """Execute ``test_erf``."""
    x = jnp.array([0.0, 1.0, -1.0])
    res = _erf(x)
    assert res.shape == x.shape


def test_nn_tanh() -> None:
    """Execute ``test_nn_tanh``."""
    x = jnp.array([0.0, 1.0])
    assert_allclose(nn.tanh(x), [math.tanh(0.0), math.tanh(1.0)])
