"""Tests for zero_jax.scipy.sparse.linalg."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.sparse.linalg as mod
from tests.conftest import assert_allclose


def test_bicgstab() -> None:
    """Test bicgstab."""
    a = jnp.array([[4.0, 1.0], [2.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    x, info = mod.bicgstab(a, b)
    assert info is None
    assert_allclose(a @ x, b, atol=1e-4)

    x_exact, _ = mod.bicgstab(a, b, x0=x)
    assert_allclose(x_exact, x)

    x_zero, _ = mod.bicgstab(jnp.zeros((2, 2)), b)
    assert x_zero.shape == (2,)

    x_diag, _ = mod.bicgstab(jnp.eye(2) * 2.0, jnp.array([4.0, 6.0]))
    assert_allclose(x_diag, [2.0, 3.0])

    a_nil = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    x_nil, _ = mod.bicgstab(a_nil, jnp.array([1.0, 1.0]))
    assert x_nil.shape == (2,)

    x_max1, _ = mod.bicgstab(a, b, maxiter=1, tol=1e-12)
    assert x_max1.shape == (2,)


def test_cg() -> None:
    """Test cg."""
    a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    x, info = mod.cg(a, b)
    assert info is None
    assert_allclose(a @ x, b, atol=1e-4)


def test_gmres() -> None:
    """Test gmres."""
    a = jnp.array([[4.0, 1.0], [2.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    x, info = mod.gmres(a, b)
    assert info is None
    assert_allclose(a @ x, b, atol=1e-4)
