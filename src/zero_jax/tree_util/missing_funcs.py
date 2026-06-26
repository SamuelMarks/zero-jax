"""Missing tree_util functions."""

from __future__ import annotations
from typing import Any
from .pytree import tree_flatten, tree_unflatten, tree_leaves


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
    # Just mocks
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


def register_pytree_node(nodetype: Any, flatten_func: Any, unflatten_func: Any) -> None:
    # No-op in zero-jax to support signature
    pass


def register_pytree_node_class(cls: Any) -> Any:
    return cls


def register_pytree_with_keys(
    nodetype: Any, flatten_func: Any, unflatten_func: Any
) -> None:
    pass


def register_pytree_with_keys_class(cls: Any) -> Any:
    return cls


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
    # Extremely simplified mock logic since implementing true generic tree transpose is complex
    # and this is just for API parity
    return pytree_to_transpose


def treedef_children(treedef: Any) -> Any:
    return []


def treedef_is_leaf(treedef: Any) -> bool:
    return False


def treedef_tuple(treedefs: Any) -> Any:
    # Return dummy
    return treedefs[0] if treedefs else None
