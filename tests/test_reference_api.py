"""Tests for tests/test_reference_api.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp_ref
import numpy as np

import zero_jax as zjax
import zero_jax.numpy as jnp_zero
from zero_jax._types import ArrayLikeSpec


def test_api_jit(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_api_jit``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_ref.sin(x) * 2.0

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_z``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_zero.sin(x) * 2.0

    x = np.array([1.0, 2.0])
    jitted_z = zjax.jit(f_z)
    jitted_r = jax.jit(f)

    check_allclose(jitted_z(x), jitted_r(x))


def test_api_grad(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_api_grad``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_ref.sum(x * x)

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_z``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_zero.sum(x * x)

    x = np.array([1.0, 2.0, 3.0])
    grad_z = zjax.grad(f_z)(x)
    grad_r = jax.grad(f)(x)
    check_allclose(grad_z, grad_r)


def test_api_value_and_grad(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_api_value_and_grad``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_ref.sum(x * x)

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_z``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_zero.sum(x * x)

    x = np.array([1.0, 2.0])
    val_z, grad_z = zjax.value_and_grad(f_z)(x)
    val_r, grad_r = jax.value_and_grad(f)(x)
    check_allclose(val_z, val_r)
    check_allclose(grad_z, grad_r)


def test_api_vmap(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_api_vmap``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return x + 1.0

    x = np.array([1.0, 2.0])

    vmapped_z = zjax.vmap(f)
    vmapped_r = jax.vmap(f)

    check_allclose(vmapped_z(x), vmapped_r(x))


def test_api_disable_jit() -> None:
    # Mostly ensuring it doesn't crash, difficult to test side effects easily
    # without running a jitted function and checking tracing
    """Execute ``test_api_disable_jit``."""
    zjax.disable_jit()
    zjax.disable_jit(False)


def test_api_pmap(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_api_pmap``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return x + 1.0

    x = np.array([[1.0, 2.0]])
    pmapped_z = zjax.pmap(f)
    pmapped_r = jax.pmap(f)
    check_allclose(pmapped_z(x), pmapped_r(x))


def test_api_eval_shape() -> None:
    """Execute ``test_api_eval_shape``."""

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_ref.sin(x)

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_z``.

        Args:
            x: Parameter ``x``.
        """
        return jnp_zero.sin(x)

    x = np.array([1.0, 2.0])
    res_r = jax.eval_shape(f, x)
    res_z = zjax.eval_shape(f_z, x)
    assert res_z.shape == res_r.shape
