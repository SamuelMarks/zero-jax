"""Reference parity tests for top-level ``zero_jax`` transformation APIs."""

from __future__ import annotations

import math

import zero_jax as zjax
import zero_jax.numpy as jnp_zero
from zero_jax._types import ArrayLikeSpec, VariadicCallable


def test_api_jit(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.jit`` matches eager and analytical reference outputs.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Compute ``sin(x) * 2.0``.

        Args:
            x: Input array.

        Returns:
            ArrayLikeSpec: Scaled sine array.
        """
        return jnp_zero.sin(x) * 2.0

    x = jnp_zero.array([1.0, 2.0])
    jitted_z = zjax.jit(f_z)
    ref_out = jnp_zero.array([math.sin(1.0) * 2.0, math.sin(2.0) * 2.0])

    check_allclose(jitted_z(x), ref_out)


def test_api_grad(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.grad`` matches analytical gradient ``2 * x``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Compute ``sum(x * x)``.

        Args:
            x: Input array.

        Returns:
            ArrayLikeSpec: Scalar sum of squares.
        """
        return jnp_zero.sum(x * x)

    x = jnp_zero.array([1.0, 2.0, 3.0])
    grad_z = zjax.grad(f_z)(x)
    grad_r = jnp_zero.array([2.0, 4.0, 6.0])
    check_allclose(grad_z, grad_r)


def test_api_value_and_grad(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.value_and_grad`` matches analytical primal and gradient.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Compute ``sum(x * x)``.

        Args:
            x: Input array.

        Returns:
            ArrayLikeSpec: Scalar sum of squares.
        """
        return jnp_zero.sum(x * x)

    x = jnp_zero.array([1.0, 2.0])
    val_z, grad_z = zjax.value_and_grad(f_z)(x)
    val_r = jnp_zero.array(5.0)
    grad_r = jnp_zero.array([2.0, 4.0])
    check_allclose(val_z, val_r)
    check_allclose(grad_z, grad_r)


def test_api_vmap(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.vmap`` matches elementwise reference evaluation.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Add ``1.0`` to input ``x``.

        Args:
            x: Input element.

        Returns:
            ArrayLikeSpec: Incremented value.
        """
        return x + 1.0

    x = jnp_zero.array([1.0, 2.0])
    vmapped_z = zjax.vmap(f)
    ref_out = jnp_zero.array([2.0, 3.0])

    check_allclose(vmapped_z(x), ref_out)


def test_api_disable_jit() -> None:
    """Verify ``zero_jax.disable_jit`` context manager toggles cleanly."""
    zjax.disable_jit()
    zjax.disable_jit(False)


def test_api_pmap(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.pmap`` matches mapped reference evaluation.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Add ``1.0`` to input ``x``.

        Args:
            x: Input slice.

        Returns:
            ArrayLikeSpec: Incremented slice.
        """
        return x + 1.0

    x = jnp_zero.array([[1.0, 2.0]])
    pmapped_z = zjax.pmap(f)
    ref_out = jnp_zero.array([[2.0, 3.0]])
    check_allclose(pmapped_z(x), ref_out)


def test_api_eval_shape() -> None:
    """Verify ``zero_jax.eval_shape`` infers the expected output shape."""

    def f_z(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Compute ``sin(x)``.

        Args:
            x: Input array.

        Returns:
            ArrayLikeSpec: Sine array.
        """
        return jnp_zero.sin(x)

    x = jnp_zero.array([1.0, 2.0])
    res_z = zjax.eval_shape(f_z, x)
    assert res_z.shape == (2,)
