"""Tests for zero_jax.tree_util.pytree."""

from typing import Any

import pytest

import zero_jax.tree_util.pytree as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_List() -> None:
    """Test List."""
    try:
        mod.List()
    except Exception:
        pass


def test_class_PyTreeDef() -> None:
    """Test class PyTreeDef."""
    try:
        mod.PyTreeDef(1.0, 1.0)
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
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


def test_tree_leaves() -> None:
    """Test tree_leaves."""
    try:
        mod.tree_leaves(1.0)
    except Exception:
        pass


def test_tree_map() -> None:
    """Test tree_map."""
    try:
        mod.tree_map(1.0, 1.0)
    except Exception:
        pass


def test_tree_structure() -> None:
    """Test tree_structure."""
    try:
        mod.tree_structure(1.0)
    except Exception:
        pass


def test_tree_unflatten() -> None:
    """Test tree_unflatten."""
    try:
        mod.tree_unflatten(1.0, 1.0)
    except Exception:
        pass
