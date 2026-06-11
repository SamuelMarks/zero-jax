"""PyTree utilities for manipulating nested data structures."""

from typing import Any
import ml_switcheroo

from .pytree import (
    tree_flatten,
    tree_unflatten,
    PyTreeDef,
    tree_map,
    tree_leaves,
    tree_structure,
    tree_all,
    tree_any,
)

__all__ = [
    "tree_flatten",
    "tree_unflatten",
    "PyTreeDef",
    "tree_map",
    "tree_leaves",
    "tree_structure",
    "tree_all",
    "tree_any",
]
