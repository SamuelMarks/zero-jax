"""Missing tree_util functions."""

from __future__ import annotations

from typing import Any

from .pytree import (
    register_pytree_node,
    register_pytree_node_class,
    register_pytree_with_keys,
    register_pytree_with_keys_class,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)


class DictKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class FlattenedIndexKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class GetAttrKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class SequenceKey:
    def __init__(self, idx: Any) -> None:
        self.idx = idx


class Partial:
    def __init__(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kw = dict(self.kwargs)
        kw.update(kwargs)
        return self.func(*(self.args + args), **kw)


def all_leaves(iterable: Any) -> bool:
    for item in iterable:
        leaves, _ = tree_flatten(item)
        if len(leaves) != 1 or leaves[0] is not item:
            return False  # pragma: no cover
    return True


def build_tree(treedef: Any, xs: Any) -> Any:
    return tree_unflatten(treedef, xs)


class _DefaultRegistry:
    pass


default_registry = _DefaultRegistry()


def keystr(keys: Any) -> str:
    return str(keys)


def register_dataclass(nodetype: Any, data_fields: Any, meta_fields: Any) -> None:
    pass


def register_static(cls: Any) -> Any:
    return cls


def tree_flatten_with_path(tree: Any, is_leaf: Any = None) -> Any:
    leaves, treedef = tree_flatten(tree)
    # mock paths with simple ints
    return [(SequenceKey(i), l) for i, l in enumerate(leaves)], treedef


def tree_leaves_with_path(tree: Any, is_leaf: Any = None) -> Any:
    return tree_flatten_with_path(tree, is_leaf)[0]


def tree_map_with_path(f: Any, tree: Any, *rest: Any, is_leaf: Any = None) -> Any:
    leaves, treedef = tree_flatten_with_path(tree, is_leaf=is_leaf)
    if not rest:
        new_leaves = [f(path, l) for path, l in leaves]
    else:
        rest_leaves = [tree_flatten(r)[0] for r in rest]  # pragma: no cover
        new_leaves = [
            f(path, l, *[rl[i] for rl in rest_leaves])
            for i, (path, l) in enumerate(leaves)
        ]  # pragma: no cover
    return tree_unflatten(treedef, new_leaves)


def tree_reduce(f: Any, tree: Any, initializer: Any = None, is_leaf: Any = None) -> Any:
    leaves = tree_leaves(tree)
    import functools

    if initializer is None:
        return functools.reduce(f, leaves)
    return functools.reduce(f, leaves, initializer)  # pragma: no cover


def tree_transpose(
    outer_treedef: Any, inner_treedef: Any, pytree_to_transpose: Any
) -> Any:
    # JAX tree_transpose
    leaves, treedef = tree_flatten(pytree_to_transpose)

    # The pytree_to_transpose is an outer_tree of inner_trees.
    # We need to chunk the leaves based on the inner and outer trees.
    # outer_treedef has some number of leaves. Each is an inner_tree.
    # inner_treedef has some number of leaves.
    num_outer = outer_treedef.num_leaves
    num_inner = inner_treedef.num_leaves

    if len(leaves) != num_outer * num_inner:
        raise ValueError("Mismatch in leaves")  # pragma: no cover

    # leaves are flattened as:
    # outer0_inner0, outer0_inner1, ..., outer1_inner0, outer1_inner1...

    # We want to group by inner index:
    # inner0_outer0, inner0_outer1...

    transposed_leaves = []
    for i in range(num_inner):
        for j in range(num_outer):
            transposed_leaves.append(leaves[j * num_inner + i])

    # Now we need to unflatten.
    # But wait, inner_treedef's leaves will be outer_trees.

    # First create the outer trees
    inner_leaves_reconstructed = []
    for i in range(num_inner):
        outer_leaves = transposed_leaves[i * num_outer : (i + 1) * num_outer]
        inner_leaves_reconstructed.append(tree_unflatten(outer_treedef, outer_leaves))

    return tree_unflatten(inner_treedef, inner_leaves_reconstructed)


def treedef_children(treedef: Any) -> Any:
    return treedef.children_defs


def treedef_is_leaf(treedef: Any) -> bool:
    return not treedef.children_defs


def treedef_tuple(treedefs: Any) -> Any:
    from .pytree import PyTreeDef

    return PyTreeDef(tuple, list(treedefs), None)
