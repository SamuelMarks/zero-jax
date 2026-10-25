"""Missing tree_util functions."""

from __future__ import annotations

import dataclasses
import functools

from zero_jax._types import ArrayLikeSpec, ConfigValue

from .pytree import (
    PyTreeDef,
    _HasFieldsProtocol,
    _is_namedtuple_instance,
    _pytree_registry,
    _pytree_with_keys_registry,
    register_pytree_node,
    register_pytree_node_class,
    register_pytree_with_keys,
    register_pytree_with_keys_class,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)


class DictKey:
    """Key path entry representing a dictionary lookup key."""

    def __init__(self, key: ArrayLikeSpec) -> None:
        """Initialize ``DictKey``.

        Args:
            key: Dictionary key value.
        """
        self.key = key

    def __eq__(self, other: ConfigValue) -> bool:
        """Check equality with another key entry.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if both are ``DictKey`` with equal keys.
        """
        return isinstance(other, DictKey) and self.key == other.key

    def __hash__(self) -> int:
        """Return hash of the ``DictKey``.

        Returns:
            int: Hash value.
        """
        return hash(("DictKey", self.key))

    def __repr__(self) -> str:
        """Return string representation in JAX key path format.

        Returns:
            str: Formatted key string.
        """
        return f"[{self.key!r}]"


class SequenceKey:
    """Key path entry representing an integer index in a tuple or list."""

    def __init__(self, idx: ArrayLikeSpec) -> None:
        """Initialize ``SequenceKey``.

        Args:
            idx: Integer sequence index.
        """
        self.idx = idx

    def __eq__(self, other: ConfigValue) -> bool:
        """Check equality with another key entry.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if both are ``SequenceKey`` with equal indices.
        """
        return isinstance(other, SequenceKey) and self.idx == other.idx

    def __hash__(self) -> int:
        """Return hash of the ``SequenceKey``.

        Returns:
            int: Hash value.
        """
        return hash(("SequenceKey", self.idx))

    def __repr__(self) -> str:
        """Return string representation in JAX key path format.

        Returns:
            str: Formatted index string.
        """
        return f"[{self.idx!r}]"


class GetAttrKey:
    """Key path entry representing an attribute access on a dataclass or custom node."""

    def __init__(self, name: ArrayLikeSpec) -> None:
        """Initialize ``GetAttrKey``.

        Args:
            name: Attribute name string.
        """
        self.name = name
        self.key = name

    def __eq__(self, other: ConfigValue) -> bool:
        """Check equality with another key entry.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if both are ``GetAttrKey`` with equal attribute names.
        """
        return isinstance(other, GetAttrKey) and self.name == other.name

    def __hash__(self) -> int:
        """Return hash of the ``GetAttrKey``.

        Returns:
            int: Hash value.
        """
        return hash(("GetAttrKey", self.name))

    def __repr__(self) -> str:
        """Return string representation in JAX key path format.

        Returns:
            str: Formatted attribute string.
        """
        return f".{self.name}"


class FlattenedIndexKey:
    """Key path entry representing a flattened index in a custom PyTree node."""

    def __init__(self, key: ArrayLikeSpec) -> None:
        """Initialize ``FlattenedIndexKey``.

        Args:
            key: Flattened index integer.
        """
        self.key = key

    def __eq__(self, other: ConfigValue) -> bool:
        """Check equality with another key entry.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if both are ``FlattenedIndexKey`` with equal keys.
        """
        return isinstance(other, FlattenedIndexKey) and self.key == other.key

    def __hash__(self) -> int:
        """Return hash of the ``FlattenedIndexKey``.

        Returns:
            int: Hash value.
        """
        return hash(("FlattenedIndexKey", self.key))

    def __repr__(self) -> str:
        """Return string representation in JAX key path format.

        Returns:
            str: Formatted index string.
        """
        return f"[<flat index {self.key!r}>]"


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
        obj_map = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else {}
        key_children = [(GetAttrKey(f), obj_map[f]) for f in resolved_data if f in obj_map]
        meta = tuple((f, obj_map[f]) for f in resolved_meta if f in obj_map)
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


def _flatten_with_path_recursive(
    tree: ArrayLikeSpec,
    prefix: tuple[ArrayLikeSpec, ...],
    is_leaf: ArrayLikeSpec = None,
) -> tuple[list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]], PyTreeDef]:
    """Recursively flatten a PyTree while recording exact key paths to each leaf.

    Args:
        tree: Subtree to flatten.
        prefix: Accumulated key path tuple.
        is_leaf: Optional leaf predicate.

    Returns:
        tuple[list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]], PyTreeDef]: Path-leaf pairs and ``PyTreeDef``.
    """
    if is_leaf is not None and callable(is_leaf) and is_leaf(tree):
        return [(prefix, tree)], PyTreeDef(type(None), [])

    node_type = type(tree)
    if node_type in _pytree_with_keys_registry:
        flatten_with_keys, _ = _pytree_with_keys_registry[node_type]
        key_children, aux = flatten_with_keys(tree)
        path_leaves: list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]] = []
        children_defs: list[PyTreeDef] = []
        for idx, item in enumerate(key_children):
            if isinstance(item, tuple) and len(item) == 2:
                k, child = item
            else:
                k, child = FlattenedIndexKey(idx), item
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, k), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs, metadata=aux)

    if node_type in _pytree_registry:
        flatten_func, _ = _pytree_registry[node_type]
        children, aux = flatten_func(tree)
        path_leaves = []
        children_defs = []
        for idx, child in enumerate(children):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, FlattenedIndexKey(idx)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs, metadata=aux)

    if _is_namedtuple_instance(tree) and isinstance(tree, _HasFieldsProtocol):
        path_leaves = []
        children_defs = []
        for field_name, child in zip(tree._fields, tree):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, GetAttrKey(field_name)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs)

    if isinstance(tree, tuple):
        path_leaves = []
        children_defs = []
        for idx, child in enumerate(tree):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, SequenceKey(idx)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(tuple, children_defs)

    if isinstance(tree, list):
        path_leaves = []
        children_defs = []
        for idx, child in enumerate(tree):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, SequenceKey(idx)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(list, children_defs)

    if isinstance(tree, dict):
        path_leaves = []
        children_defs = []
        keys = sorted(tree.keys())
        for k in keys:
            c_leaves, c_def = _flatten_with_path_recursive(tree[k], (*prefix, DictKey(k)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(dict, children_defs, metadata=keys)

    return [(prefix, tree)], PyTreeDef(type(None), [])


def tree_flatten_with_path(
    tree: ArrayLikeSpec,
    is_leaf: ArrayLikeSpec = None,
) -> tuple[list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]], PyTreeDef]:
    """Flatten a PyTree into ``(key_path, leaf)`` pairs and a :class:`PyTreeDef`.

    Args:
        tree: PyTree to flatten.
        is_leaf: Optional leaf predicate.

    Returns:
        tuple[list[tuple[tuple[ArrayLikeSpec, ...], ArrayLikeSpec]], PyTreeDef]: ``(path_leaves, treedef)``.
    """
    return _flatten_with_path_recursive(tree, (), is_leaf=is_leaf)


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
