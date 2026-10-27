"""Reference parity tests for ``zero_jax.tree_util``."""

from __future__ import annotations

import zero_jax.tree_util as tree_zero
from zero_jax._types import VariadicCallable


def test_tree_flatten(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.tree_util.tree_flatten`` flattens and unflattens a nested dict/list PyTree.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    tree = {"a": 1, "b": [2, 3]}
    leaves_z, treedef_z = tree_zero.tree_flatten(tree)
    leaves_r = [1, 2, 3]

    check_allclose(leaves_z, leaves_r)
    check_allclose(tree_zero.tree_unflatten(treedef_z, leaves_z), tree)


def test_tree_unflatten(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.tree_util.tree_unflatten`` reconstructs a nested list/tuple PyTree.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    tree = [1, (2, 3)]
    leaves_z, treedef_z = tree_zero.tree_flatten(tree)

    check_allclose(tree_zero.tree_unflatten(treedef_z, leaves_z), tree)


def test_tree_map(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.tree_util.tree_map`` applies a function across all PyTree leaves.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    tree = {"a": 1, "b": [2, 3]}
    res_z = tree_zero.tree_map(lambda x: x * 2, tree)
    res_r = {"a": 2, "b": [4, 6]}
    check_allclose(res_z, res_r)


def test_tree_leaves(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.tree_util.tree_leaves`` extracts all leaves in deterministic order.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    tree = {"a": 1, "b": [2, 3]}
    res_z = tree_zero.tree_leaves(tree)
    res_r = [1, 2, 3]
    check_allclose(res_z, res_r)


def test_tree_structure() -> None:
    """Verify ``zero_jax.tree_util.tree_structure`` reports expected leaf and node counts."""
    tree = {"a": 1, "b": [2, 3]}
    struct_z = tree_zero.tree_structure(tree)
    assert struct_z.num_leaves == 3
    assert struct_z.num_nodes == 5


def test_tree_all_skip() -> None:
    """Verify ``zero_jax.tree_util.tree_all`` reduces boolean leaves with logical AND."""
    tree = {"a": True, "b": [True, False]}
    assert tree_zero.tree_all(tree) is False
    assert tree_zero.tree_all({"a": True, "b": [True, True]}) is True
