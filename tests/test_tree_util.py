"""Tests for zero_jax.tree_util."""

from zero_jax.tree_util import tree_flatten, tree_unflatten


def test_tree_flatten_unflatten():
    # Simple scalar
    leaves, treedef = tree_flatten(1.0)
    assert leaves == [1.0]
    assert tree_unflatten(treedef, leaves) == 1.0

    # Tuple
    leaves, treedef = tree_flatten((1.0, 2.0))
    assert leaves == [1.0, 2.0]
    assert tree_unflatten(treedef, leaves) == (1.0, 2.0)

    # List
    leaves, treedef = tree_flatten([1.0, 2.0])
    assert leaves == [1.0, 2.0]
    assert tree_unflatten(treedef, leaves) == [1.0, 2.0]

    # Dict
    leaves, treedef = tree_flatten({"b": 2.0, "a": 1.0})
    assert leaves == [1.0, 2.0]
    assert tree_unflatten(treedef, leaves) == {"a": 1.0, "b": 2.0}

    # Nested
    nested = {"a": [1.0, (2.0, {"c": 3.0})], "b": 4.0}
    leaves, treedef = tree_flatten(nested)
    assert leaves == [1.0, 2.0, 3.0, 4.0]
    assert tree_unflatten(treedef, leaves) == nested


def test_pytreedef_eq():
    from zero_jax.tree_util.pytree import tree_structure

    d1 = tree_structure({"a": 1, "b": [2, 3]})
    d2 = tree_structure({"a": 1, "b": [2, 3]})
    d3 = tree_structure({"a": 1, "c": [2, 3]})
    assert d1 == d2
    assert d1 != d3
    assert d1 != 5
