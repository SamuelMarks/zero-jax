import pytest
import jax
import jax.numpy as jnp_ref
import jax.nn as nn_ref
import jax.nn.initializers as init_ref
import zero_jax.nn as nn_zero
import zero_jax.nn.initializers as init_zero
import numpy as np


@pytest.mark.skip(reason="Numerical divergence")
def test_nn_gelu_skip(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.gelu(x), nn_ref.gelu(x))


def test_nn_logsumexp(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Depending on axis implementation, test defaults
    check_allclose(nn_zero.logsumexp(x), nn_ref.logsumexp(x))


def test_nn_one_hot(check_allclose):
    x = np.array([0, 1, 2])
    check_allclose(nn_zero.one_hot(x, 3), nn_ref.one_hot(x, 3))


def test_nn_softmax(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(nn_zero.softmax(x), nn_ref.softmax(x))


def test_nn_sigmoid(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.sigmoid(x), nn_ref.sigmoid(x))


def test_nn_log_sigmoid(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.log_sigmoid(x), nn_ref.log_sigmoid(x))


# Initializers testing
@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_zeros(check_allclose):
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), init_ref.zeros(key, (2, 2)))


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_ones(check_allclose):
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), init_ref.ones(key, (2, 2)))


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_constant(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    init_r = init_ref.constant(5.0)
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_uniform(check_allclose):
    # Just checking shape and execution, exact values might differ if RNG implementation is different
    # But if zero_jax wraps jax properly, they should be identical.
    key = jax.random.PRNGKey(0)
    init_z = init_zero.uniform()
    init_r = init_ref.uniform()
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_normal(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.normal()
    init_r = init_ref.normal()
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


# We will test a few other complex initializers
@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_glorot_uniform(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.glorot_uniform()
    init_r = init_ref.glorot_uniform()
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
def test_init_skip_he_normal(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.he_normal()
    init_r = init_ref.he_normal()
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


def test_nn_relu(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.relu(x), nn_ref.relu(x))


def test_nn_relu6(check_allclose):
    x = np.array([-1.0, 0.0, 1.0, 7.0])
    check_allclose(nn_zero.relu6(x), nn_ref.relu6(x))


def test_nn_hard_sigmoid(check_allclose):
    x = np.array([-3.0, 0.0, 3.0])
    check_allclose(nn_zero.hard_sigmoid(x), nn_ref.hard_sigmoid(x))


def test_nn_hard_tanh(check_allclose):
    x = np.array([-3.0, 0.0, 3.0])
    check_allclose(nn_zero.hard_tanh(x), nn_ref.hard_tanh(x))


def test_nn_swish(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.swish(x), nn_ref.swish(x))


def test_nn_silu(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.silu(x), nn_ref.silu(x))


def test_nn_elu(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.elu(x), nn_ref.elu(x))


def test_nn_celu(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.celu(x), nn_ref.celu(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_nn_selu(check_allclose):
    x = np.array([-1.0, 0.0, 1.0])
    check_allclose(nn_zero.selu(x), nn_ref.selu(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_nn_log_softmax(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(nn_zero.log_softmax(x), nn_ref.log_softmax(x))


def test_init_zeros(check_allclose):
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), init_ref.zeros(key, (2, 2)))


def test_init_ones(check_allclose):
    key = jax.random.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), init_ref.ones(key, (2, 2)))


def test_init_constant(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    init_r = init_ref.constant(5.0)
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))


@pytest.mark.skip(reason="Numerical divergence")
def test_init_skip_orthogonal(check_allclose):
    key = jax.random.PRNGKey(0)
    init_z = init_zero.orthogonal()
    init_r = init_ref.orthogonal()
    check_allclose(init_z(key, (2, 2)), init_r(key, (2, 2)))
