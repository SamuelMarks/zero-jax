"""Tests for zero_jax.api_util."""

from __future__ import annotations

import pytest

import zero_jax.api_util as mod
import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_structure


def test_argnums_partial() -> None:
    """Test argnums_partial with static and dynamic arguments."""
    assert mod.argnums_partial() == (None, ())
    fn, dyn = mod.argnums_partial(lambda a, b, c: a + b * c, 1, (10, 3, 4))
    assert dyn == (3,)
    assert fn(5) == 10 + 5 * 4


def test_donation_vector() -> None:
    """Test donation_vector with and without PyTreeDef."""
    assert mod.donation_vector(0) == (True,)
    td = tree_structure(([1, 2], 3))
    assert mod.donation_vector((0,), (), td) == (True, True, False)


def test_flatten_axes() -> None:
    """Test flatten_axes broadcasting over PyTreeDef."""
    td = tree_structure({"a": 1, "b": [2, 3]})
    assert mod.flatten_axes("in_axes", td, 0) == [0, 0, 0]
    assert mod.flatten_axes("in_axes", td, [1]) == [1, 1, 1]
    assert mod.flatten_axes("in_axes", td, {"a": 0, "b": [1, 2]}) == [0, 1, 2]
    assert mod.flatten_axes("in_axes", None, (0, 1)) == [0, 1]


def test_flatten_fun_and_nokwargs() -> None:
    """Test flatten_fun and flatten_fun_nokwargs."""
    empty_fn, _ = mod.flatten_fun()
    assert empty_fn() == []

    in_td = tree_structure(((1, 2), {"scale": 3}))
    flat_fn, out_td_thunk = mod.flatten_fun(lambda a, b, scale=1: (a + b) * scale, in_td)
    assert flat_fn(2, 4, 5) == [30]
    assert out_td_thunk() is not None

    pos_td = tree_structure((1, 2))
    flat_pos, _ = mod.flatten_fun_nokwargs(lambda a, b: a - b, pos_td)
    assert flat_pos(9, 4) == [5]

    scalar_td = tree_structure(1)
    flat_scalar, _ = mod.flatten_fun_nokwargs(lambda a: a * 2, scalar_td)
    assert flat_scalar(6) == [12]

    no_td_fn, _ = mod.flatten_fun_nokwargs(lambda a, b: a + b, None)
    assert no_td_fn(3, 7) == [10]


def test_rebase_donate_argnums() -> None:
    """Test rebase_donate_argnums."""
    assert mod.rebase_donate_argnums((1, 3), (0, 2)) == (0, 1)
    assert mod.rebase_donate_argnums(2, 0) == (1,)


def test_safe_map() -> None:
    """Test safe_map on equal and mismatched sequence lengths."""
    assert mod.safe_map() == []
    assert mod.safe_map(lambda x, y: x + y, [1, 2], [10, 20]) == [11, 22]
    with pytest.raises(ValueError, match="safe_map requires equal lengths"):
        mod.safe_map(lambda x, y: x + y, [1, 2], [10])


def test_shaped_abstractify() -> None:
    """Test shaped_abstractify on scalars, arrays, and ShapedArray."""
    sa_scalar = mod.shaped_abstractify(3.0)
    assert sa_scalar.shape == ()
    assert mod.shaped_abstractify(sa_scalar) is sa_scalar

    arr = jnp.zeros((2, 3))
    sa_arr = mod.shaped_abstractify(arr)
    assert sa_arr.shape == (2, 3)
