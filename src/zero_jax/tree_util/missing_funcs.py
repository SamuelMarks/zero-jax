"""Missing tree_util functions."""

from __future__ import annotations

import dataclasses
import functools
import operator

from zero_jax._types import ArrayLikeSpec, ConfigValue

from .pytree import (
    DictKey,
    FlattenedIndexKey,
    GetAttrKey,
    PyTreeDef,
    SequenceKey,
    _flatten_with_path_recursive,
    _HasFieldsProtocol,
    _is_namedtuple_instance,
    _pytree_registry,
    _pytree_with_keys_registry,
    register_pytree_node,
    register_pytree_node_class,
    register_pytree_with_keys,
    register_pytree_with_keys_class,
    tree_flatten,
    tree_flatten_with_path,
    tree_leaves,
    tree_unflatten,
)


class Partial:
    """A version of ``functools.partial`` that is a registered PyTree node."""

    def __init__(self, func: ArrayLikeSpec, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> None:
        """Initialize ``Partial``.

        Args:
            func: Callable to partially apply.
            *args: Bound positional arguments (treated as PyTree children).
            **kwargs: Bound keyword arguments (treated as PyTree children).
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke ``self.func`` with bound and call-time arguments.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ArrayLikeSpec: Result of calling ``self.func``.
        """
        kw = dict(self.kwargs)
        kw.update(kwargs)
        return self.func(*(self.args + args), **kw)

    def __eq__(self, other: ConfigValue) -> bool:
        """Compare two ``Partial`` instances.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if function, args, and kwargs match.
        """
        return isinstance(other, Partial) and self.func == other.func and self.args == other.args and self.kwargs == other.kwargs


register_pytree_node(
    Partial,
    lambda p: ((p.args, p.kwargs), p.func),
    lambda func, children: Partial(func, *children[0], **children[1]),
)


def all_leaves(iterable: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> bool:
    """Test whether all elements in the given iterable are PyTree leaves.

    Args:
        iterable: Iterable of potential leaves.
        is_leaf: Optional leaf predicate.

    Returns:
        bool: ``True`` if every item in ``iterable`` is a single leaf.
    """
    for item in iterable:
        leaves, treedef = tree_flatten(item, is_leaf=is_leaf)
        if treedef.node_type is not type(None) or len(leaves) != 1 or leaves[0] is not item:
            return False
    return True


def build_tree(treedef: PyTreeDef, xs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reconstruct a PyTree from ``treedef`` and nested iterable ``xs``.

    Args:
        treedef: Target ``PyTreeDef`` structure.
        xs: Sequence of leaf or child values.

    Returns:
        ArrayLikeSpec: Reconstructed PyTree.
    """
    flat_leaves, _ = tree_flatten(xs)
    return tree_unflatten(treedef, flat_leaves)


class _DefaultRegistry:
    """Default PyTree node registry container."""


default_registry = _DefaultRegistry()


def keystr(keys: ArrayLikeSpec, *, simple: bool = False, separator: str = "") -> str:
    """Format a key path tuple into a human-readable string.

    Args:
        keys: Sequence of key entries (`DictKey`, `SequenceKey`, `GetAttrKey`, `FlattenedIndexKey`).
        simple: Whether to use simplified key formatting.
        separator: Optional separator between path components.

    Returns:
        str: Formatted key path string.
    """
    if isinstance(keys, (tuple, list)):
        if simple:
            parts: list[str] = []
            for k in keys:
                if isinstance(k, DictKey):
                    parts.append(str(k.key))
                elif isinstance(k, SequenceKey):
                    parts.append(str(k.idx))
                elif isinstance(k, GetAttrKey):
                    parts.append(str(k.name))
                elif isinstance(k, FlattenedIndexKey):
                    parts.append(str(k.key))
                else:
                    parts.append(str(k))
            return (separator or "/").join(parts)
        return separator.join(repr(k) for k in keys)
    return str(keys)


def register_dataclass(
    nodetype: type,
    data_fields: ArrayLikeSpec = None,
    meta_fields: ArrayLikeSpec = None,
) -> type:
    """Register a dataclass type as a PyTree node with specified data and metadata fields.

    Args:
        nodetype: Dataclass or Python class to register.
        data_fields: Optional sequence of field names containing PyTree child leaves.
        meta_fields: Optional sequence of field names containing static metadata.

    Returns:
        type: The registered ``nodetype``.
    """
    if not isinstance(nodetype, type):
        return nodetype

    dc_fields = [f.name for f in dataclasses.fields(nodetype)] if dataclasses.is_dataclass(nodetype) else []
    resolved_data = list(data_fields) if data_fields is not None else dc_fields
    resolved_meta = list(meta_fields) if meta_fields is not None else []

    def _flatten_with_keys(obj: ArrayLikeSpec) -> tuple[list[tuple[GetAttrKey, ArrayLikeSpec]], tuple[tuple[str, ArrayLikeSpec], ...]]:
        """Flatten dataclass instance with ``GetAttrKey`` entries.

        Args:
            obj: Dataclass instance.

        Returns:
            tuple[list[tuple[GetAttrKey, ArrayLikeSpec]], tuple[tuple[str, ArrayLikeSpec], ...]]: Keyed children and static metadata.
        """
        key_children = [(GetAttrKey(f), operator.attrgetter(f)(obj)) for f in resolved_data]
        meta = tuple((f, operator.attrgetter(f)(obj)) for f in resolved_meta)
        return key_children, meta

    def _unflatten(meta: tuple[tuple[str, ArrayLikeSpec], ...], children: list[ArrayLikeSpec]) -> ArrayLikeSpec:
        """Reconstruct dataclass instance from static metadata and dynamic children.

        Args:
            meta: Static field pairs.
            children: Dynamic field values.

        Returns:
            ArrayLikeSpec: Reconstructed dataclass instance.
        """
        kwargs: dict[str, ArrayLikeSpec] = dict(meta)
        for f, val in zip(resolved_data, children):
            kwargs[f] = val
        return nodetype(**kwargs)

    register_pytree_with_keys(nodetype, _flatten_with_keys, _unflatten)
    return nodetype


def register_static(cls: type) -> type:
    """Register ``cls`` as a static PyTree node with zero leaves.

    Args:
        cls: Class to register as static.

    Returns:
        type: The registered class ``cls``.
    """
    if isinstance(cls, type):
        register_pytree_node(cls, lambda obj: ((), obj), lambda obj, _: obj)
    return cls


def tree_leaves_with_path(
    tree: ArrayLikeSpec,
    is_leaf: ArrayLikeSpec = None,
) -> list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]]:
    """Return a list of ``(key_path, leaf)`` pairs for all leaves in ``tree``.

    Args:
        tree: PyTree to inspect.
        is_leaf: Optional leaf predicate.

    Returns:
        list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]]: Path-leaf pairs.
    """
    return tree_flatten_with_path(tree, is_leaf=is_leaf)[0]


def tree_map_with_path(
    f: ArrayLikeSpec,
    tree: ArrayLikeSpec,
    *rest: ArrayLikeSpec,
    is_leaf: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Map a function ``f(key_path, leaf, *rest_leaves)`` over a PyTree.

    Args:
        f: Function accepting ``(key_path, leaf, ...)``.
        tree: Primary PyTree.
        *rest: Additional PyTrees of identical structure.
        is_leaf: Optional leaf predicate.

    Returns:
        ArrayLikeSpec: Mapped PyTree.
    """
    leaves, treedef = tree_flatten_with_path(tree, is_leaf=is_leaf)
    if not rest:
        new_leaves = [f(path, leaf) for path, leaf in leaves]
    else:
        rest_leaves = [tree_flatten(r, is_leaf=is_leaf)[0] for r in rest]
        new_leaves = [f(path, leaf, *[rl[i] for rl in rest_leaves]) for i, (path, leaf) in enumerate(leaves)]
    return tree_unflatten(treedef, new_leaves)


def tree_reduce(
    f: ArrayLikeSpec,
    tree: ArrayLikeSpec,
    initializer: ArrayLikeSpec = None,
    is_leaf: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Reduce a binary function ``f`` over the leaves of ``tree``.

    Args:
        f: Binary reduction function ``f(accum, leaf)``.
        tree: PyTree to reduce.
        initializer: Optional initial accumulator value.
        is_leaf: Optional leaf predicate.

    Returns:
        ArrayLikeSpec: Reduced result.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf)
    if initializer is None:
        return functools.reduce(f, leaves)
    return functools.reduce(f, leaves, initializer)


def tree_transpose(
    outer_treedef: PyTreeDef,
    inner_treedef: PyTreeDef,
    pytree_to_transpose: ArrayLikeSpec,
) -> ArrayLikeSpec:
    """Transpose a nested PyTree having structure ``outer_treedef`` of ``inner_treedef`` into ``inner_treedef`` of ``outer_treedef``.

    Args:
        outer_treedef: Outer ``PyTreeDef`` structure.
        inner_treedef: Inner ``PyTreeDef`` structure.
        pytree_to_transpose: Nested PyTree to transpose.

    Returns:
        ArrayLikeSpec: Transposed PyTree.

    Raises:
        ValueError: If leaf count does not match ``outer_treedef.num_leaves * inner_treedef.num_leaves``.
    """
    leaves, _ = tree_flatten(pytree_to_transpose)
    num_outer = outer_treedef.num_leaves
    num_inner = inner_treedef.num_leaves

    if len(leaves) != num_outer * num_inner:
        raise ValueError("Mismatch in leaves")

    transposed_leaves = []
    for i in range(num_inner):
        for j in range(num_outer):
            transposed_leaves.append(leaves[j * num_inner + i])

    inner_leaves_reconstructed = []
    for i in range(num_inner):
        outer_leaves = transposed_leaves[i * num_outer : (i + 1) * num_outer]
        inner_leaves_reconstructed.append(tree_unflatten(outer_treedef, outer_leaves))

    return tree_unflatten(inner_treedef, inner_leaves_reconstructed)


def treedef_children(treedef: PyTreeDef) -> list[PyTreeDef]:
    """Return the child ``PyTreeDef`` list of ``treedef``.

    Args:
        treedef: Structure definition.

    Returns:
        list[PyTreeDef]: Child definitions.
    """
    return treedef.children_defs


def treedef_is_leaf(treedef: PyTreeDef) -> bool:
    """Return ``True`` if ``treedef`` represents a single leaf node.

    Args:
        treedef: Structure definition.

    Returns:
        bool: Whether ``treedef`` has no children and is not an empty container.
    """
    return not treedef.children_defs and treedef.node_type is type(None)


def treedef_tuple(treedefs: ArrayLikeSpec) -> PyTreeDef:
    """Combine a sequence of ``PyTreeDef`` objects into a tuple ``PyTreeDef``.

    Args:
        treedefs: Iterable of child ``PyTreeDef`` objects.

    Returns:
        PyTreeDef: Tuple ``PyTreeDef`` wrapping ``treedefs``.
    """
    return PyTreeDef(tuple, list(treedefs), None)


__all__ = [
    "DictKey",
    "FlattenedIndexKey",
    "GetAttrKey",
    "Partial",
    "SequenceKey",
    "_DefaultRegistry",
    "_flatten_with_path_recursive",
    "all_leaves",
    "build_tree",
    "default_registry",
    "keystr",
    "register_dataclass",
    "register_pytree_node",
    "register_pytree_node_class",
    "register_pytree_with_keys",
    "register_pytree_with_keys_class",
    "register_static",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_map_with_path",
    "tree_reduce",
    "tree_transpose",
    "tree_unflatten",
    "treedef_children",
    "treedef_is_leaf",
    "treedef_tuple",
]
