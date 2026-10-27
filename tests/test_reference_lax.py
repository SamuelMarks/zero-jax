"""Reference parity tests for ``zero_jax.lax`` primitives."""

from __future__ import annotations

import zero_jax.lax as lax_zero
import zero_jax.numpy as jnp_zero
from zero_jax._types import ArrayLikeSpec, VariadicCallable


def test_lax_cond(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.cond`` branches conditionally.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def true_fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Add ``1.0`` to ``x``.

        Args:
            x: Input value.

        Returns:
            ArrayLikeSpec: Incremented value.
        """
        return x + 1.0

    def false_fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Subtract ``1.0`` from ``x``.

        Args:
            x: Input value.

        Returns:
            ArrayLikeSpec: Decremented value.
        """
        return x - 1.0

    val = jnp_zero.array(5.0)
    check_allclose(
        lax_zero.cond(True, true_fn, false_fn, val),
        jnp_zero.array(6.0),
    )
    check_allclose(
        lax_zero.cond(False, true_fn, false_fn, val),
        jnp_zero.array(4.0),
    )


def test_lax_scan(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.scan`` accumulates carry and stacked outputs.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """

    def f(carry: ArrayLikeSpec, x: ArrayLikeSpec) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
        """Advance scan carry and compute step output.

        Args:
            carry: Current scan carry.
            x: Current sequence element.

        Returns:
            tuple[ArrayLikeSpec, ArrayLikeSpec]: Updated carry and step output.
        """
        return carry + x, carry * x

    xs = jnp_zero.array([1.0, 2.0, 3.0])
    init = jnp_zero.array(0.0)

    carry_z, y_z = lax_zero.scan(f, init, xs)
    carry_r = jnp_zero.array(6.0)
    y_r = jnp_zero.array([0.0, 2.0, 9.0])

    check_allclose(carry_z, carry_r)
    check_allclose(y_z, y_r)


def test_lax_stop_gradient(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.stop_gradient`` returns primal values unchanged.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(lax_zero.stop_gradient(x), jnp_zero.array([1.0, 2.0]))


def test_lax_add(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.add`` computes elementwise addition.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([1.0]), jnp_zero.array([2.0])
    check_allclose(lax_zero.add(x, y), jnp_zero.array([3.0]))


def test_lax_sub(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.sub`` computes elementwise subtraction.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(lax_zero.sub(x, y), jnp_zero.array([3.0]))


def test_lax_mul(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.mul`` computes elementwise multiplication.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(lax_zero.mul(x, y), jnp_zero.array([10.0]))


def test_lax_div(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.div`` computes elementwise division.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(lax_zero.div(x, y), jnp_zero.array([2.5]))


def test_lax_broadcast(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.broadcast`` prepends leading dimensions.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(
        lax_zero.broadcast(x, (2,)),
        jnp_zero.array([[1.0, 2.0], [1.0, 2.0]]),
    )


def test_lax_broadcast_in_dim(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.broadcast_in_dim`` broadcasts along specified dimensions.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(
        lax_zero.broadcast_in_dim(x, (2, 2), (0,)),
        jnp_zero.array([[1.0, 1.0], [2.0, 2.0]]),
    )


def test_lax_reshape(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.reshape`` reshapes an array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0, 3.0, 4.0])
    check_allclose(
        lax_zero.reshape(x, (2, 2)),
        jnp_zero.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_lax_transpose(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.transpose`` permutes array axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(
        lax_zero.transpose(x, (1, 0)),
        jnp_zero.array([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_lax_slice(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.slice`` extracts a static slice.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.arange(10.0)
    check_allclose(
        lax_zero.slice(x, (2,), (8,)),
        jnp_zero.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    )


def test_lax_dynamic_slice(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.dynamic_slice`` extracts a dynamic slice.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.arange(10.0)
    check_allclose(
        lax_zero.dynamic_slice(x, (2,), (3,)),
        jnp_zero.array([2.0, 3.0, 4.0]),
    )


def test_lax_dynamic_update_slice(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.dynamic_update_slice`` updates a slice at a dynamic start index.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.zeros(10)
    update = jnp_zero.ones(3)
    check_allclose(
        lax_zero.dynamic_update_slice(x, update, (2,)),
        jnp_zero.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_lax_gather(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.gather`` gathers elements using ``GatherDimensionNumbers``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    operand = jnp_zero.array([10.0, 20.0, 30.0, 40.0])
    indices = jnp_zero.array([[1], [3]])
    dnums = lax_zero.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    check_allclose(
        lax_zero.gather(operand, indices, dnums, (1,)),
        jnp_zero.array([20.0, 40.0]),
    )


def test_lax_scatter(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.scatter`` scatters updates using ``ScatterDimensionNumbers``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    operand = jnp_zero.array([1.0, 2.0, 3.0, 4.0])
    indices = jnp_zero.array([[1], [3]])
    updates = jnp_zero.array([10.0, 30.0])
    dnums = lax_zero.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    check_allclose(
        lax_zero.scatter(operand, indices, updates, dnums),
        jnp_zero.array([1.0, 10.0, 3.0, 30.0]),
    )


def test_lax_reduce_sum(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.reduce`` reduces an axis with ``lax.add``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(
        lax_zero.reduce(x, 0.0, lax_zero.add, (0,)),
        jnp_zero.array([4.0, 6.0]),
    )


def test_lax_select(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.select`` selects elements from true/false branches.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    cond = jnp_zero.array([True, False])
    x = jnp_zero.array([1.0, 2.0])
    y = jnp_zero.array([3.0, 4.0])
    check_allclose(lax_zero.select(cond, x, y), jnp_zero.array([1.0, 4.0]))


def test_lax_clamp(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.lax.clamp`` clamps elements between bounds.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 5.0, 10.0])
    min_val = jnp_zero.array(2.0)
    max_val = jnp_zero.array(8.0)
    check_allclose(lax_zero.clamp(min_val, x, max_val), jnp_zero.array([2.0, 5.0, 8.0]))
