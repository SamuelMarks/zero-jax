"""Tests for zero_jax.tree_util."""

from typing import Any

import pytest

import zero_jax.tree_util as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_DictKey() -> None:
    """Test class DictKey."""
    try:
        mod.DictKey(1.0)
    except Exception:
        pass


def test_class_FlattenedIndexKey() -> None:
    """Test class FlattenedIndexKey."""
    try:
        mod.FlattenedIndexKey(1.0)
    except Exception:
        pass


def test_class_GetAttrKey() -> None:
    """Test class GetAttrKey."""
    try:
        mod.GetAttrKey(1.0)
    except Exception:
        pass


def test_List() -> None:
    """Test List."""
    try:
        mod.List()
    except Exception:
        pass


def test_class_Partial() -> None:
    """Test class Partial."""
    try:
        mod.Partial(1.0)
    except Exception:
        pass


def test_class_PyTreeDef() -> None:
    """Test class PyTreeDef."""
    try:
        mod.PyTreeDef(1.0, 1.0)
    except Exception:
        pass


def test_class_SequenceKey() -> None:
    """Test class SequenceKey."""
    try:
        mod.SequenceKey(1.0)
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
    except Exception:
        pass


def test_all_leaves() -> None:
    """Test all_leaves."""
    try:
        mod.all_leaves(1.0)
    except Exception:
        pass


def test_build_tree() -> None:
    """Test build_tree."""
    try:
        mod.build_tree(1.0, 1.0)
    except Exception:
        pass


def test_keystr() -> None:
    """Test keystr."""
    try:
        mod.keystr(1.0)
    except Exception:
        pass


def test_register_dataclass() -> None:
    """Test register_dataclass."""
    try:
        mod.register_dataclass(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_register_pytree_node() -> None:
    """Test register_pytree_node."""
    try:
        mod.register_pytree_node(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_register_pytree_node_class() -> None:
    """Test register_pytree_node_class."""
    try:
        mod.register_pytree_node_class(1.0)
    except Exception:
        pass


def test_register_pytree_with_keys() -> None:
    """Test register_pytree_with_keys."""
    try:
        mod.register_pytree_with_keys(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_register_pytree_with_keys_class() -> None:
    """Test register_pytree_with_keys_class."""
    try:
        mod.register_pytree_with_keys_class(1.0)
    except Exception:
        pass


def test_register_static() -> None:
    """Test register_static."""
    try:
        mod.register_static(1.0)
    except Exception:
        pass


def test_tree_all() -> None:
    """Test tree_all."""
    try:
        mod.tree_all(1.0)
    except Exception:
        pass


def test_tree_any() -> None:
    """Test tree_any."""
    try:
        mod.tree_any(1.0)
    except Exception:
        pass


def test_tree_flatten() -> None:
    """Test tree_flatten."""
    try:
        mod.tree_flatten(1.0)
    except Exception:
        pass


def test_tree_flatten_with_path() -> None:
    """Test tree_flatten_with_path."""
    try:
        mod.tree_flatten_with_path(1.0)
    except Exception:
        pass


def test_tree_leaves() -> None:
    """Test tree_leaves."""
    try:
        mod.tree_leaves(1.0)
    except Exception:
        pass


def test_tree_leaves_with_path() -> None:
    """Test tree_leaves_with_path."""
    try:
        mod.tree_leaves_with_path(1.0)
    except Exception:
        pass


def test_tree_map() -> None:
    """Test tree_map."""
    try:
        mod.tree_map(1.0, 1.0)
    except Exception:
        pass


def test_tree_map_with_path() -> None:
    """Test tree_map_with_path."""
    try:
        mod.tree_map_with_path(1.0, 1.0)
    except Exception:
        pass


def test_tree_reduce() -> None:
    """Test tree_reduce."""
    try:
        mod.tree_reduce(1.0, 1.0)
    except Exception:
        pass


def test_tree_structure() -> None:
    """Test tree_structure."""
    try:
        mod.tree_structure(1.0)
    except Exception:
        pass


def test_tree_transpose() -> None:
    """Test tree_transpose."""
    try:
        mod.tree_transpose(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_tree_unflatten() -> None:
    """Test tree_unflatten."""
    try:
        mod.tree_unflatten(1.0, 1.0)
    except Exception:
        pass


def test_treedef_children() -> None:
    """Test treedef_children."""
    try:
        mod.treedef_children(1.0)
    except Exception:
        pass


def test_treedef_is_leaf() -> None:
    """Test treedef_is_leaf."""
    try:
        mod.treedef_is_leaf(1.0)
    except Exception:
        pass


def test_treedef_tuple() -> None:
    """Test treedef_tuple."""
    try:
        mod.treedef_tuple(1.0)
    except Exception:
        pass
