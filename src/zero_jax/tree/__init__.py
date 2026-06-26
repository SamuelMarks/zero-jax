"""Frontend API routing for jax.tree."""

from typing import Any
import ml_switcheroo_compiler.tree_util as _tu


def all(tree: Any) -> bool:
    """Checks if all leaves of a PyTree are truthy."""
    return _tu.tree_all(tree)


def flatten(tree: Any) -> Any:
    """Flattens a PyTree into a list of leaves and a treedef."""
    return _tu.tree_flatten(tree)


def leaves(tree: Any) -> Any:
    """Gets the leaves of a PyTree."""
    return _tu.tree_leaves(tree)


def map(f: Any, tree: Any, *rest: Any) -> Any:
    """Maps a function over the leaves of a PyTree."""
    return _tu.tree_map(f, tree, *rest)


def reduce(f: Any, tree: Any, initializer: Any = None) -> Any:
    """Reduces a PyTree by applying a function over its leaves."""
    return _tu.tree_reduce(f, tree, initializer)


def structure(tree: Any) -> Any:
    """Gets the structure of a PyTree."""
    return _tu.tree_structure(tree)


def transpose(outer_treedef: Any, inner_treedef: Any, pytree_to_transpose: Any) -> Any:
    """Transposes a PyTree of PyTrees."""
    return _tu.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


def unflatten(treedef: Any, leaves: Any) -> Any:
    """Reconstructs a PyTree from a treedef and a list of leaves."""
    return _tu.tree_unflatten(treedef, leaves)


__all__ = [
    "all",
    "flatten",
    "leaves",
    "map",
    "reduce",
    "structure",
    "transpose",
    "unflatten",
]
