"""Tests for zero_jax.nn.missing_funcs."""

from __future__ import annotations

import math

import zero_jax.nn.missing_funcs as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_glu() -> None:
    """Test glu."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    out = mod.glu(x)
    assert out.shape == (2,)


def test_hard_silu() -> None:
    """Test hard_silu."""
    out = mod.hard_silu(jnp.array([-4.0, 0.0, 4.0]))
    assert_allclose(out, [0.0, 0.0, 4.0])


def test_hard_swish() -> None:
    """Test hard_swish."""
    out = mod.hard_swish(jnp.array([-4.0, 0.0, 4.0]))
    assert_allclose(out, [0.0, 0.0, 4.0])
    assert_allclose(mod.identity(jnp.array([1, 2])), [1, 2])


def test_leaky_relu() -> None:
    """Test leaky_relu."""
    out = mod.leaky_relu(jnp.array([-2.0, 3.0]), negative_slope=0.1)
    assert_allclose(out, [-0.2, 3.0])


def test_mish() -> None:
    """Test mish."""
    out = mod.mish(jnp.array([0.0]))
    assert_allclose(out, [0.0])


def test_soft_sign() -> None:
    """Test soft_sign."""
    out = mod.soft_sign(jnp.array([-1.0, 0.0, 1.0]))
    assert_allclose(out, [-0.5, 0.0, 0.5])


def test_softplus() -> None:
    """Test softplus."""
    out = mod.softplus(jnp.array([0.0]))
    assert_allclose(out, [math.log(2.0)], rtol=1e-5)


def test_sparse_plus() -> None:
    """Test sparse_plus."""
    out = mod.sparse_plus(jnp.array([-2.0, 0.0, 2.0]))
    assert_allclose(out, [0.0, 0.25, 2.0])


def test_sparse_sigmoid() -> None:
    """Test sparse_sigmoid."""
    out = mod.sparse_sigmoid(jnp.array([-2.0, 0.0, 2.0]))
    assert_allclose(out, [0.0, 0.5, 1.0])


def test_squareplus() -> None:
    """Test squareplus."""
    out = mod.squareplus(jnp.array([0.0]), b=4.0)
    assert_allclose(out, [1.0])


def test_standardize() -> None:
    """Test standardize and normalize."""
    x = jnp.array([1.0, 2.0, 3.0])
    out = mod.standardize(x)
    assert_allclose(jnp.mean(out), 0.0, atol=1e-5)
    norm_out = mod.normalize(jnp.array([3.0, 4.0]))
    assert_allclose(norm_out, [0.6, 0.8], atol=1e-4)
    norm_std = mod.normalize(x, mean=2.0, variance=1.0)
    assert_allclose(norm_std, [-1.0, 0.0, 1.0], atol=1e-4)
    assert_allclose(mod.squareplus(jnp.array([0]), b=4.0), [1.0])
