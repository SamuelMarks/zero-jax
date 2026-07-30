"""Tests for zero_jax.tree module."""

import pytest

import zero_jax.tree as ztree


def test_all() -> None:
    """Test all."""
    assert ztree.all({"a": True, "b": [1, True]})
    assert not ztree.all({"a": True, "b": [0, True]})


def test_flatten() -> None:
    """Test flatten."""
    leaves, treedef = ztree.flatten({"a": 1, "b": [2, 3]})
    assert leaves == [1, 2, 3]


def test_leaves() -> None:
    """Test leaves."""
    leaves = ztree.leaves({"a": 1, "b": [2, 3]})
    assert leaves == [1, 2, 3]


def test_map() -> None:
    """Test map."""
    tree = {"a": 1, "b": [2, 3]}
    res = ztree.map(lambda x: x * 2, tree)
    assert res == {"a": 2, "b": [4, 6]}


def test_reduce() -> None:
    """Test reduce."""
    tree = {"a": 1, "b": [2, 3]}
    res = ztree.reduce(lambda x, y: x + y, tree)
    assert res == 6


def test_structure() -> None:
    """Test structure."""
    treedef = ztree.structure({"a": 1, "b": [2, 3]})
    assert treedef is not None


def test_transpose() -> None:
    """Test transpose."""
    tree = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    outer_def = ztree.structure([0, 0])
    inner_def = ztree.structure({"a": 0, "b": 0})
    transposed = ztree.transpose(outer_def, inner_def, tree)
    assert transposed == {"a": [1, 3], "b": [2, 4]}


def test_unflatten() -> None:
    """Test unflatten."""
    treedef = ztree.structure({"a": 1, "b": [2, 3]})
    res = ztree.unflatten(treedef, [1, 2, 3])
    assert res == {"a": 1, "b": [2, 3]}
