"""Frontend API routing for jax.tree."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.tree_util.missing_funcs import tree_reduce, tree_transpose
from zero_jax.tree_util.pytree import (
    PyTreeDef,
    tree_all,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)


def all(tree: ArrayLikeSpec, *, is_leaf: ArrayLikeSpec = None) -> bool:
    """Check whether all leaves of a PyTree are truthy.

    Args:
        tree: Input PyTree.
        is_leaf: Optional leaf predicate.

    Returns:
        bool: ``True`` if all leaves evaluate to ``True``.
    """
    return tree_all(tree, is_leaf=is_leaf)


def flatten(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> tuple[list[ArrayLikeSpec], PyTreeDef]:
    """Flatten a PyTree into a list of leaves and a ``PyTreeDef``.

    Args:
        tree: Input PyTree.
        is_leaf: Optional leaf predicate.

    Returns:
        tuple[list[ArrayLikeSpec], PyTreeDef]: ``(leaves, treedef)``.
    """
    return tree_flatten(tree, is_leaf=is_leaf)


def leaves(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> list[ArrayLikeSpec]:
    """Return the flat list of leaves of a PyTree.

    Args:
        tree: Input PyTree.
        is_leaf: Optional leaf predicate.

    Returns:
        list[ArrayLikeSpec]: Leaves of ``tree``.
    """
    return tree_leaves(tree, is_leaf=is_leaf)


def map(f: ArrayLikeSpec, tree: ArrayLikeSpec, *rest: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Map a multi-argument function ``f`` over the leaves of one or more PyTrees.

    Args:
        f: Callable to apply to leaves.
        tree: Primary PyTree.
        *rest: Additional PyTrees with matching structure.
        is_leaf: Optional leaf predicate.

    Returns:
        ArrayLikeSpec: New PyTree with mapped leaves.
    """
    return tree_map(f, tree, *rest, is_leaf=is_leaf)


def reduce(
    f: ArrayLikeSpec,
    tree: ArrayLikeSpec,
    initializer: ArrayLikeSpec = None,
    is_leaf: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Reduce a binary function ``f`` over the leaves of a PyTree.

    Args:
        f: Binary reduction callable.
        tree: Input PyTree.
        initializer: Optional initial accumulator value.
        is_leaf: Optional leaf predicate.

    Returns:
        ArrayLikeSpec: Reduced value.
    """
    return tree_reduce(f, tree, initializer=initializer, is_leaf=is_leaf)


def structure(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> PyTreeDef:
    """Return the :class:`PyTreeDef` structure of a PyTree.

    Args:
        tree: Input PyTree.
        is_leaf: Optional leaf predicate.

    Returns:
        PyTreeDef: Structure definition.
    """
    return tree_structure(tree, is_leaf=is_leaf)


def transpose(
    outer_treedef: PyTreeDef,
    inner_treedef: PyTreeDef,
    pytree_to_transpose: ArrayLikeSpec,
) -> ArrayLikeSpec:
    """Transpose a PyTree with structure ``outer_treedef`` of ``inner_treedef``.

    Args:
        outer_treedef: Outer ``PyTreeDef`` structure.
        inner_treedef: Inner ``PyTreeDef`` structure.
        pytree_to_transpose: Nested PyTree to transpose.

    Returns:
        ArrayLikeSpec: Transposed PyTree.
    """
    return tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


def unflatten(treedef: PyTreeDef, leaves: list[ArrayLikeSpec]) -> ArrayLikeSpec:
    """Reconstruct a PyTree from its ``treedef`` and ``leaves``.

    Args:
        treedef: Structure definition.
        leaves: List of leaf values.

    Returns:
        ArrayLikeSpec: Reconstructed PyTree.
    """
    return tree_unflatten(treedef, leaves)


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
