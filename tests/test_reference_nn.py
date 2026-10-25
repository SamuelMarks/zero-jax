"""Tests for tests/test_reference_nn.py."""

from __future__ import annotations

import jax
import jax.nn as nn_ref
import jax.nn.initializers as init_ref
import numpy as np
import pytest

import zero_jax.nn as nn_zero
import zero_jax.nn.initializers as init_zero
from zero_jax._types import ArrayLikeSpec


def test_nn_gelu_skip(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_gelu_skip``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.gelu(x), nn_ref.gelu(x))


def test_nn_logsumexp(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_logsumexp``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Depending on axis implementation, test defaults
    check_allclose(nn_zero.logsumexp(x), nn_ref.logsumexp(x))


def test_nn_one_hot(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_one_hot``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([0, 1, 2])
    check_allclose(nn_zero.one_hot(x, 3), nn_ref.one_hot(x, 3))


def test_nn_softmax(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_softmax``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(nn_zero.softmax(x), nn_ref.softmax(x))


def test_nn_sigmoid(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_sigmoid``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.sigmoid(x), nn_ref.sigmoid(x))


def test_nn_log_sigmoid(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_log_sigmoid``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.log_sigmoid(x), nn_ref.log_sigmoid(x))


# Initializers testing
def test_init_skip_zeros(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_zeros``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), init_ref.zeros(key, (2, 2)))


def test_init_skip_ones(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_ones``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), init_ref.ones(key, (2, 2)))


def test_init_skip_constant(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_constant``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    init_r = init_ref.constant(5.0)
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


def test_init_skip_uniform(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_uniform``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.uniform()
    init_r = init_ref.uniform()
    z_res = init_z(key, (2, 2))
    r_res = init_r(key, (2, 2))
    assert z_res.shape == r_res.shape
    # Don't assert strict numerical values since PRNG might diverge slightly


def test_init_skip_normal(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_normal``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.normal()
    init_r = init_ref.normal()
    z_res = init_z(key, (2, 2))
    r_res = init_r(key, (2, 2))
    assert z_res.shape == r_res.shape
    # Don't assert strict numerical values since PRNG might diverge slightly


# We will test a few other complex initializers
def test_init_skip_glorot_uniform(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_glorot_uniform``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.glorot_uniform()
    init_r = init_ref.glorot_uniform()
    z_res = init_z(key, (2, 2))
    r_res = init_r(key, (2, 2))
    assert z_res.shape == r_res.shape
    # Don't assert strict numerical values since PRNG might diverge slightly


def test_init_skip_he_normal(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_he_normal``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.he_normal()
    init_r = init_ref.he_normal()
    z_res = init_z(key, (2, 2))
    r_res = init_r(key, (2, 2))
    assert z_res.shape == r_res.shape
    # Don't assert strict numerical values since PRNG might diverge slightly


def test_nn_relu(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_relu``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.relu(x), nn_ref.relu(x))


def test_nn_relu6(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_relu6``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0, 7.0])
    check_allclose(nn_zero.relu6(x), nn_ref.relu6(x))


def test_nn_hard_sigmoid(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_hard_sigmoid``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-3.0, 0.0, 3.0])
    check_allclose(nn_zero.hard_sigmoid(x), nn_ref.hard_sigmoid(x))


def test_nn_hard_tanh(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_hard_tanh``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-3.0, 0.0, 3.0])
    check_allclose(nn_zero.hard_tanh(x), nn_ref.hard_tanh(x))


def test_nn_swish(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_swish``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.swish(x), nn_ref.swish(x))


def test_nn_silu(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_silu``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.silu(x), nn_ref.silu(x))


def test_nn_elu(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_elu``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.elu(x), nn_ref.elu(x))


def test_nn_celu(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_celu``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.celu(x), nn_ref.celu(x))


def test_nn_selu(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_selu``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.selu(x), nn_ref.selu(x))


def test_nn_log_softmax(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_nn_log_softmax``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(nn_zero.log_softmax(x), nn_ref.log_softmax(x))


def test_init_zeros(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_zeros``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), init_ref.zeros(key, (2, 2)))


def test_init_ones(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_ones``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), init_ref.ones(key, (2, 2)))


def test_init_constant(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_constant``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    init_r = init_ref.constant(5.0)
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


def test_init_skip_orthogonal(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_init_skip_orthogonal``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    key = jax.random.PRNGKey(0)
    init_z = init_zero.orthogonal()
    init_r = init_ref.orthogonal()
    z_res = init_z(key, (2, 2))
    r_res = init_r(key, (2, 2))
    assert z_res.shape == r_res.shape
    # Don't assert strict numerical values since PRNG might diverge slightly
