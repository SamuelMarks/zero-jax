"""Reference parity tests for ``zero_jax.nn`` and ``zero_jax.nn.initializers``."""

from __future__ import annotations

import math

import pytest

import zero_jax.nn as nn_zero
import zero_jax.nn.initializers as init_zero
import zero_jax.numpy as jnp_zero
import zero_jax.random as rand_zero
from zero_jax._types import VariadicCallable


def _sigmoid_scalar(v: float) -> float:
    """Compute scalar logistic sigmoid ``1 / (1 + exp(-v))``.

    Args:
        v: Input float.

    Returns:
        float: Sigmoid value.
    """
    return 1.0 / (1.0 + math.exp(-v))


def test_nn_gelu_skip(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.gelu`` against analytical approximate GELU.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    ref = [0.5 * v * (1.0 + math.tanh(sqrt_2_over_pi * (v + 0.044715 * (v**3)))) for v in vals]
    check_allclose(nn_zero.gelu(x), jnp_zero.array(ref), rtol=1e-4, atol=1e-4)


def test_nn_logsumexp(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.logsumexp`` against analytical log-sum-exp.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    expected = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0) + math.exp(4.0))
    check_allclose(nn_zero.logsumexp(x), jnp_zero.array(expected))


def test_nn_one_hot(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.one_hot`` produces expected one-hot matrix.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([0, 1, 2])
    expected = jnp_zero.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    check_allclose(nn_zero.one_hot(x, 3), expected)


def test_nn_softmax(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.softmax`` against analytical row-wise softmax.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    e1, e2 = math.exp(1.0), math.exp(2.0)
    e3, e4 = math.exp(3.0), math.exp(4.0)
    ref = jnp_zero.array([[e1 / (e1 + e2), e2 / (e1 + e2)], [e3 / (e3 + e4), e4 / (e3 + e4)]])
    check_allclose(nn_zero.softmax(x), ref)


def test_nn_sigmoid(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.sigmoid`` against ``_sigmoid_scalar``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(nn_zero.sigmoid(x), jnp_zero.array([_sigmoid_scalar(v) for v in vals]))


def test_nn_log_sigmoid(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.log_sigmoid`` against ``log(sigmoid(x))``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.log_sigmoid(x),
        jnp_zero.array([math.log(_sigmoid_scalar(v)) for v in vals]),
    )


def test_init_skip_zeros(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.zeros`` returns an all-zeros array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), jnp_zero.zeros((2, 2)))


def test_init_skip_ones(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.ones`` returns an all-ones array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), jnp_zero.ones((2, 2)))


def test_init_skip_constant(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.constant`` fills with the specified value.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    check_allclose(init_z(key, (2, 2)), jnp_zero.full((2, 2), 5.0))


def test_init_skip_uniform(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.uniform`` returns an array of expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.uniform()
    z_res = init_z(key, (2, 2))
    assert z_res.shape == (2, 2)


def test_init_skip_normal(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.normal`` returns an array of expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.normal()
    z_res = init_z(key, (2, 2))
    assert z_res.shape == (2, 2)


def test_init_skip_glorot_uniform(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.glorot_uniform`` returns an array of expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.glorot_uniform()
    z_res = init_z(key, (2, 2))
    assert z_res.shape == (2, 2)


def test_init_skip_he_normal(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.he_normal`` returns an array of expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.he_normal()
    z_res = init_z(key, (2, 2))
    assert z_res.shape == (2, 2)


def test_nn_relu(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.relu`` against ``max(0, x)``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(nn_zero.relu(x), jnp_zero.array([max(0.0, v) for v in vals]))


def test_nn_relu6(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.relu6`` against ``min(max(0, x), 6)``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0, 7.0]
    x = jnp_zero.array(vals)
    check_allclose(nn_zero.relu6(x), jnp_zero.array([min(max(0.0, v), 6.0) for v in vals]))


def test_nn_hard_sigmoid(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.hard_sigmoid`` against ``relu6(x + 3) / 6``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-3.0, 0.0, 3.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.hard_sigmoid(x),
        jnp_zero.array([min(max(0.0, v + 3.0), 6.0) / 6.0 for v in vals]),
    )


def test_nn_hard_tanh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.hard_tanh`` clamps to ``[-1, 1]``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-3.0, 0.0, 3.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.hard_tanh(x),
        jnp_zero.array([max(-1.0, min(1.0, v)) for v in vals]),
    )


def test_nn_swish(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.swish`` against ``x * sigmoid(x)``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.swish(x),
        jnp_zero.array([v * _sigmoid_scalar(v) for v in vals]),
    )


def test_nn_silu(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.silu`` against ``x * sigmoid(x)``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.silu(x),
        jnp_zero.array([v * _sigmoid_scalar(v) for v in vals]),
    )


def test_nn_elu(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.elu`` against analytical ELU.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.elu(x),
        jnp_zero.array([v if v > 0.0 else math.expm1(v) for v in vals]),
    )


def test_nn_celu(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.celu`` against analytical CELU.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(
        nn_zero.celu(x),
        jnp_zero.array([max(0.0, v) + math.expm1(min(0.0, v)) for v in vals]),
    )


def test_nn_selu(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.selu`` against analytical SELU constants.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [-1.0, 0.0, 1.0]
    x = jnp_zero.array(vals)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    ref = [scale * (v if v > 0.0 else alpha * math.expm1(v)) for v in vals]
    check_allclose(nn_zero.selu(x), jnp_zero.array(ref))


def test_nn_log_softmax(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.log_softmax`` against analytical log-softmax.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    lse0 = math.log(math.exp(1.0) + math.exp(2.0))
    lse1 = math.log(math.exp(3.0) + math.exp(4.0))
    ref = jnp_zero.array([[1.0 - lse0, 2.0 - lse0], [3.0 - lse1, 4.0 - lse1]])
    check_allclose(nn_zero.log_softmax(x), ref)


def test_init_zeros(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.zeros`` returns a zero matrix.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    check_allclose(init_zero.zeros(key, (2, 2)), jnp_zero.zeros((2, 2)))


def test_init_ones(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.ones`` returns a ones matrix.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    check_allclose(init_zero.ones(key, (2, 2)), jnp_zero.ones((2, 2)))


def test_init_constant(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.constant`` fills with constant scalar.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.constant(5.0)
    check_allclose(init_z(key, (2, 2)), jnp_zero.full((2, 2), 5.0))


def test_init_skip_orthogonal(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.nn.initializers.orthogonal`` produces the requested matrix shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    pytest.mark.parametrize("shape", [(2, 2)])
    key = rand_zero.PRNGKey(0)
    init_z = init_zero.orthogonal()
    z_res = init_z(key, (2, 2))
    assert z_res.shape == (2, 2)
