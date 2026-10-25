"""Tests for tests/test_reference_lax.py."""

from __future__ import annotations

import jax.lax as lax_ref
import numpy as np

import zero_jax.lax as lax_zero
from zero_jax._types import ArrayLikeSpec


def test_lax_cond(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_cond``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def true_fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``true_fn``.

        Args:
            x: Parameter ``x``.
        """
        return x + 1.0

    def false_fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``false_fn``.

        Args:
            x: Parameter ``x``.
        """
        return x - 1.0

    val = np.array(5.0)
    check_allclose(
        lax_zero.cond(True, true_fn, false_fn, val),
        lax_ref.cond(True, true_fn, false_fn, val),
    )
    check_allclose(
        lax_zero.cond(False, true_fn, false_fn, val),
        lax_ref.cond(False, true_fn, false_fn, val),
    )


def test_lax_scan(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_scan``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """

    def f(carry: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``.

        Args:
            carry: Parameter ``carry``.
            x: Parameter ``x``.
        """
        return carry + x, carry * x

    xs = np.array([1.0, 2.0, 3.0])
    init = np.array(0.0)

    carry_z, y_z = lax_zero.scan(f, init, xs)
    carry_r, y_r = lax_ref.scan(f, init, xs)

    check_allclose(carry_z, carry_r)
    check_allclose(y_z, y_r)


def test_lax_stop_gradient(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_stop_gradient``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([1.0, 2.0])
    check_allclose(lax_zero.stop_gradient(x), lax_ref.stop_gradient(x))


def test_lax_add(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_add``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x, y = np.array([1.0]), np.array([2.0])
    check_allclose(lax_zero.add(x, y), lax_ref.add(x, y))


def test_lax_sub(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_sub``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(lax_zero.sub(x, y), lax_ref.sub(x, y))


def test_lax_mul(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_mul``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(lax_zero.mul(x, y), lax_ref.mul(x, y))


def test_lax_div(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_div``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(lax_zero.div(x, y), lax_ref.div(x, y))


def test_lax_broadcast(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_broadcast``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([1.0, 2.0])
    check_allclose(lax_zero.broadcast(x, (2,)), lax_ref.broadcast(x, (2,)))


def test_lax_broadcast_in_dim(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_broadcast_in_dim``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([1.0, 2.0])
    check_allclose(
        lax_zero.broadcast_in_dim(x, (2, 2), (0,)),
        lax_ref.broadcast_in_dim(x, (2, 2), (0,)),
    )


def test_lax_reshape(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_reshape``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    check_allclose(lax_zero.reshape(x, (2, 2)), lax_ref.reshape(x, (2, 2)))


def test_lax_transpose(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_transpose``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(lax_zero.transpose(x, (1, 0)), lax_ref.transpose(x, (1, 0)))


def test_lax_slice(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_slice``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.arange(10.0)
    check_allclose(lax_zero.slice(x, (2,), (8,)), lax_ref.slice(x, (2,), (8,)))


def test_lax_dynamic_slice(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_dynamic_slice``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.arange(10.0)
    check_allclose(lax_zero.dynamic_slice(x, (2,), (3,)), lax_ref.dynamic_slice(x, (2,), (3,)))


def test_lax_dynamic_update_slice(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_dynamic_update_slice``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.zeros(10)
    update = np.ones(3)
    check_allclose(
        lax_zero.dynamic_update_slice(x, update, (2,)),
        lax_ref.dynamic_update_slice(x, update, (2,)),
    )


def test_lax_gather(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_gather``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    operand = np.array([10.0, 20.0, 30.0, 40.0])
    indices = np.array([[1], [3]])
    dnums = lax_zero.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    dnums_ref = lax_ref.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    check_allclose(
        lax_zero.gather(operand, indices, dnums, (1,)),
        lax_ref.gather(operand, indices, dnums_ref, (1,)),
    )


def test_lax_scatter(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_scatter``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    operand = np.array([1.0, 2.0, 3.0, 4.0])
    indices = np.array([[1], [3]])
    updates = np.array([10.0, 30.0])
    dnums = lax_zero.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    dnums_ref = lax_ref.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    check_allclose(
        lax_zero.scatter(operand, indices, updates, dnums),
        lax_ref.scatter(operand, indices, updates, dnums_ref),
    )


def test_lax_reduce_sum(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_reduce_sum``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(
        lax_zero.reduce(x, 0.0, lax_zero.add, (0,)),
        lax_ref.reduce(x, 0.0, lax_ref.add, (0,)),
    )


def test_lax_select(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_select``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    cond = np.array([True, False])
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    check_allclose(lax_zero.select(cond, x, y), lax_ref.select(cond, x, y))


def test_lax_clamp(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_lax_clamp``.

    Args:
        check_allclose: Parameter ``check_allclose``.
    """
    x = np.array([1.0, 5.0, 10.0])
    min_val = np.array(2.0)
    max_val = np.array(8.0)
    check_allclose(lax_zero.clamp(min_val, x, max_val), lax_ref.clamp(min_val, x, max_val))
