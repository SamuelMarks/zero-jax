"""PyTree manipulation utilities."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from zero_jax._types import (
    ArrayLikeSpec,
    ConfigValue,
    CustomStateMappingProtocol,
    CustomVariableProtocol,
    MappableDataclassProtocol,
    PyTree,
    RecordProtocol,
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


@runtime_checkable
class _HasFieldsProtocol(Protocol):
    """Protocol for ``namedtuple`` instances exposing ``_fields``."""

    _fields: tuple[str, ...]


@runtime_checkable
class _TreeFlattenWithKeysProtocol(Protocol):
    """Protocol for classes implementing ``tree_flatten_with_keys`` and ``tree_unflatten``."""

    def tree_flatten_with_keys(self) -> tuple[Iterable[ArrayLikeSpec], ArrayLikeSpec]:
        """Flatten instance into keyed child pairs and auxiliary metadata.

        Returns:
            tuple[Iterable[ArrayLikeSpec], ArrayLikeSpec]: Keyed children and auxiliary data.
        """
        ...

    @classmethod
    def tree_unflatten(cls, aux: ArrayLikeSpec, children: Sequence[ArrayLikeSpec]) -> PyTree:
        """Reconstruct instance from auxiliary metadata and flattened children.

        Args:
            aux: Auxiliary metadata.
            children: Reconstructed child values.

        Returns:
            PyTree: Reconstructed instance.
        """
        ...


class FlattenFunc(Protocol):
    """Protocol for PyTree flattening callables ``node -> (children_iterable, aux_data)``."""

    def __call__(self, node: PyTree) -> tuple[Iterable[PyTree], ArrayLikeSpec]:
        """Flatten ``node`` into an iterable of child subtrees and auxiliary metadata.

        Args:
            node: Custom PyTree node instance.

        Returns:
            tuple[Iterable[PyTree], ArrayLikeSpec]: ``(children, aux_data)``.
        """
        ...


class UnflattenFunc(Protocol):
    """Protocol for PyTree unflattening callables ``(aux_data, children_list) -> node``."""

    def __call__(self, aux: ArrayLikeSpec, children: list[PyTree]) -> PyTree:
        """Reconstruct a custom PyTree node from auxiliary metadata and child subtrees.

        Args:
            aux: Auxiliary metadata.
            children: List of reconstructed child subtrees.

        Returns:
            PyTree: Reconstructed node.
        """
        ...


_pytree_registry: dict[type, tuple[FlattenFunc, UnflattenFunc]] = {}
_pytree_with_keys_registry: dict[type, tuple[FlattenFunc, UnflattenFunc]] = {}


def _is_namedtuple_instance(x: PyTree) -> bool:
    """Check whether ``x`` is a ``namedtuple`` instance.

    Args:
        x: Candidate object.

    Returns:
        bool: ``True`` if ``x`` is a tuple subclass defining ``_fields``.
    """
    return isinstance(x, tuple) and type(x) is not tuple and isinstance(x, _HasFieldsProtocol)


def _extract_key_child_pairs(pairs: Iterable[ArrayLikeSpec]) -> list[tuple[ArrayLikeSpec, ArrayLikeSpec]]:
    """Normalize an iterable of keyed or plain children into ``(key, child)`` pairs.

    Args:
        pairs: Iterable of ``(key, child)`` tuples or plain child values.

    Returns:
        list[tuple[ArrayLikeSpec, ArrayLikeSpec]]: Normalized ``(key, child)`` pairs.
    """
    normalized: list[tuple[ArrayLikeSpec, ArrayLikeSpec]] = []
    for idx, item in enumerate(pairs):
        if isinstance(item, tuple) and len(item) == 2:
            normalized.append((item[0], item[1]))
        else:
            normalized.append((FlattenedIndexKey(idx), item))
    return normalized


def register_pytree_node(nodetype: type, flatten_func: FlattenFunc, unflatten_func: UnflattenFunc) -> None:
    """Register a custom container type with the PyTree registry.

    Args:
        nodetype: Python type to register as an internal PyTree node.
        flatten_func: Function mapping ``node -> (children_iterable, aux_data)``.
        unflatten_func: Function mapping ``(aux_data, children_list) -> node``.
    """
    _pytree_registry[nodetype] = (flatten_func, unflatten_func)


def register_pytree_node_class(cls: type) -> type:
    """Register a class that defines ``tree_flatten`` and ``tree_unflatten`` methods.

    Args:
        cls: Class to register.

    Returns:
        type: The registered class ``cls``.
    """
    register_pytree_node(
        cls,
        lambda obj: obj.tree_flatten(),
        lambda aux, children: cls.tree_unflatten(aux, children),
    )
    return cls


def register_pytree_with_keys(
    nodetype: type,
    flatten_with_keys: FlattenFunc,
    unflatten_func: UnflattenFunc,
    flatten_func: FlattenFunc | None = None,
) -> None:
    """Register a custom container type with key-aware flattening in the PyTree registry.

    Args:
        nodetype: Python type to register.
        flatten_with_keys: Function mapping ``node -> (key_child_pairs, aux_data)``.
        unflatten_func: Function mapping ``(aux_data, children) -> node``.
        flatten_func: Optional plain flatten function ``node -> (children, aux_data)``.
    """
    if flatten_func is None:

        def _plain_flatten(obj: ArrayLikeSpec) -> tuple[list[ArrayLikeSpec], ArrayLikeSpec]:
            """Extract children from key-child pairs or plain children.

            Args:
                obj: Custom PyTree node instance.

            Returns:
                tuple[list[ArrayLikeSpec], ArrayLikeSpec]: ``(children, aux)``.
            """
            pairs, aux = flatten_with_keys(obj)
            return [child for _, child in _extract_key_child_pairs(pairs)], aux

        resolved_flatten: FlattenFunc = _plain_flatten
    else:
        resolved_flatten = flatten_func

    _pytree_registry[nodetype] = (resolved_flatten, unflatten_func)
    _pytree_with_keys_registry[nodetype] = (flatten_with_keys, unflatten_func)


def register_pytree_with_keys_class(cls: type) -> type:
    """Register a class that defines ``tree_flatten_with_keys`` (or ``tree_flatten``) and ``tree_unflatten``.

    Args:
        cls: Class to register.

    Returns:
        type: The registered class ``cls``.
    """
    if issubclass(cls, _TreeFlattenWithKeysProtocol):
        register_pytree_with_keys(
            cls,
            lambda obj: obj.tree_flatten_with_keys(),
            lambda aux, children: cls.tree_unflatten(aux, children),
        )
        return cls
    return register_pytree_node_class(cls)


class PyTreeDef:
    """Represents the structural definition of a PyTree.

    Attributes:
        node_type: The container type of the node (or ``type(None)`` for a leaf).
        children_defs: Structural definitions of the child nodes.
        metadata: Auxiliary metadata required to reconstruct the node.
    """

    def __init__(self, node_type: type, children_defs: list[PyTreeDef], metadata: ArrayLikeSpec = None) -> None:
        """Initialize a ``PyTreeDef``.

        Args:
            node_type: Container type of the node.
            children_defs: List of child ``PyTreeDef`` definitions.
            metadata: Auxiliary node metadata.
        """
        self.node_type = node_type
        self.children_defs = children_defs
        self.metadata = metadata

    @property
    def num_leaves(self) -> int:
        """Return the number of leaves in the PyTree structure.

        Returns:
            int: Total number of leaf nodes.
        """
        if not self.children_defs:
            is_empty_container = (
                self.node_type in (tuple, list, dict)
                or self.node_type in _pytree_registry
                or issubclass(
                    self.node_type,
                    (
                        tuple,
                        MappableDataclassProtocol,
                        CustomStateMappingProtocol,
                        _TreeFlattenWithKeysProtocol,
                    ),
                )
            )
            return 0 if is_empty_container else 1
        return sum(c.num_leaves for c in self.children_defs)

    @property
    def num_nodes(self) -> int:
        """Return the total number of internal and leaf nodes in the PyTree structure.

        Returns:
            int: Total node count.
        """
        return 1 + sum(c.num_nodes for c in self.children_defs)

    def __eq__(self, other: PyTreeDef | ConfigValue) -> bool:
        """Compare two ``PyTreeDef`` instances for structural equality.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if structurally equal.
        """
        if not isinstance(other, PyTreeDef):
            return False
        return self.node_type == other.node_type and self.metadata == other.metadata and self.children_defs == other.children_defs


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
        for k, child in _extract_key_child_pairs(key_children):
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

    if isinstance(tree, _TreeFlattenWithKeysProtocol):
        key_children, aux = tree.tree_flatten_with_keys()
        path_leaves = []
        children_defs = []
        for k, child in _extract_key_child_pairs(key_children):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, k), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs, metadata=aux)

    if isinstance(tree, CustomVariableProtocol):
        c_leaves, c_def = _flatten_with_path_recursive(tree.value, (*prefix, GetAttrKey("value")), is_leaf=is_leaf)
        return c_leaves, PyTreeDef(node_type, [c_def], metadata=tree)

    if isinstance(tree, MappableDataclassProtocol):
        fields = tuple(tree.__chex_mappable_fields__())
        values = tuple(tree.__chex_field_values__())
        path_leaves = []
        children_defs = []
        for field_name, child in zip(fields, values):
            c_leaves, c_def = _flatten_with_path_recursive(child, (*prefix, GetAttrKey(field_name)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs, metadata=fields)

    if isinstance(tree, RecordProtocol) and dataclasses.is_dataclass(tree):
        c_leaves, c_def = _flatten_with_path_recursive(tree.data, (*prefix, GetAttrKey("data")), is_leaf=is_leaf)
        return c_leaves, PyTreeDef(node_type, [c_def], metadata=tree)

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

    if type(tree) is dict:
        path_leaves = []
        children_defs = []
        keys = sorted(tree.keys())
        for k in keys:
            c_leaves, c_def = _flatten_with_path_recursive(tree[k], (*prefix, DictKey(k)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(dict, children_defs, metadata=keys)

    if isinstance(tree, CustomStateMappingProtocol):
        path_leaves = []
        children_defs = []
        keys = sorted(tree.keys())
        for k in keys:
            c_leaves, c_def = _flatten_with_path_recursive(tree[k], (*prefix, DictKey(k)), is_leaf=is_leaf)
            path_leaves.extend(c_leaves)
            children_defs.append(c_def)
        return path_leaves, PyTreeDef(node_type, children_defs, metadata=keys)

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


def tree_flatten(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> tuple[list[ArrayLikeSpec], PyTreeDef]:
    """Flatten a PyTree into a list of leaves and a :class:`PyTreeDef` structure.

    Args:
        tree: The PyTree to flatten.
        is_leaf: Optional predicate ``callable(node) -> bool`` treating matching nodes as leaves.

    Returns:
        tuple[list[ArrayLikeSpec], PyTreeDef]: ``(leaves, treedef)``.
    """
    path_leaves, treedef = _flatten_with_path_recursive(tree, (), is_leaf=is_leaf)
    return [leaf for _, leaf in path_leaves], treedef


def _unflatten_from_iter(treedef: PyTreeDef, leaf_list: list[ArrayLikeSpec]) -> ArrayLikeSpec:
    """Recursively consume leaves from ``leaf_list`` according to ``treedef``.

    Args:
        treedef: Structure definition.
        leaf_list: Mutable list of remaining leaves.

    Returns:
        ArrayLikeSpec: Reconstructed PyTree node.
    """
    if treedef.node_type in _pytree_registry:
        _, unflatten_func = _pytree_registry[treedef.node_type]
        children = [_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs]
        return unflatten_func(treedef.metadata, children)
    if issubclass(treedef.node_type, _TreeFlattenWithKeysProtocol):
        children = [_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs]
        return treedef.node_type.tree_unflatten(treedef.metadata, children)
    if isinstance(treedef.metadata, CustomVariableProtocol):
        child_val = _unflatten_from_iter(treedef.children_defs[0], leaf_list)
        return treedef.metadata.replace(value=child_val)
    if issubclass(treedef.node_type, MappableDataclassProtocol):
        children = [_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs]
        fields = treedef.metadata if isinstance(treedef.metadata, (tuple, list)) else ()
        return treedef.node_type(**dict(zip(fields, children)))
    if isinstance(treedef.metadata, RecordProtocol):
        child_val = _unflatten_from_iter(treedef.children_defs[0], leaf_list)
        return treedef.node_type(metadata=treedef.metadata.metadata, data=child_val)
    if treedef.node_type is tuple:
        return tuple(_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs)
    if issubclass(treedef.node_type, tuple):
        return treedef.node_type(*(_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs))
    if treedef.node_type is list:
        return [_unflatten_from_iter(child_def, leaf_list) for child_def in treedef.children_defs]
    if treedef.node_type is dict:
        children_dict: dict[ArrayLikeSpec, ArrayLikeSpec] = {}
        keys = treedef.metadata if treedef.metadata is not None else []
        for k, child_def in zip(keys, treedef.children_defs):
            children_dict[k] = _unflatten_from_iter(child_def, leaf_list)
        return children_dict
    if issubclass(treedef.node_type, CustomStateMappingProtocol):
        custom_dict: dict[ArrayLikeSpec, ArrayLikeSpec] = {}
        keys = treedef.metadata if treedef.metadata is not None else []
        for k, child_def in zip(keys, treedef.children_defs):
            custom_dict[k] = _unflatten_from_iter(child_def, leaf_list)
        return treedef.node_type(custom_dict)
    return leaf_list.pop(0)


def tree_unflatten(treedef: PyTreeDef, leaves: list[ArrayLikeSpec]) -> ArrayLikeSpec:
    """Reconstruct a PyTree from its ``treedef`` structure and ``leaves`` sequence.

    Args:
        treedef: The PyTree structure definition.
        leaves: Sequence of leaf values (not mutated in-place).

    Returns:
        ArrayLikeSpec: The reconstructed PyTree.
    """
    leaf_copy = list(leaves)
    return _unflatten_from_iter(treedef, leaf_copy)


def tree_leaves(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> list[ArrayLikeSpec]:
    """Extract all leaves from a PyTree.

    Args:
        tree: The PyTree to extract leaves from.
        is_leaf: Optional leaf predicate.

    Returns:
        list[ArrayLikeSpec]: Flat list of leaf values.
    """
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf)
    return leaves


def tree_structure(tree: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> PyTreeDef:
    """Extract the :class:`PyTreeDef` structure of a PyTree.

    Args:
        tree: The PyTree to analyze.
        is_leaf: Optional leaf predicate.

    Returns:
        PyTreeDef: Structure descriptor of ``tree``.
    """
    _, treedef = tree_flatten(tree, is_leaf=is_leaf)
    return treedef


def tree_map(f: ArrayLikeSpec, tree: ArrayLikeSpec, *rest: ArrayLikeSpec, is_leaf: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Map a multi-argument function ``f`` over the corresponding leaves of one or more PyTrees.

    Args:
        f: Callable to apply to each tuple of leaves.
        tree: Primary PyTree.
        *rest: Additional PyTrees with matching structure.
        is_leaf: Optional leaf predicate.

    Returns:
        ArrayLikeSpec: New PyTree with mapped leaves.
    """
    leaves, treedef = tree_flatten(tree, is_leaf=is_leaf)
    all_leaves = [leaves]
    for r in rest:
        r_leaves, _ = tree_flatten(r, is_leaf=is_leaf)
        all_leaves.append(r_leaves)
    mapped_leaves = [f(*args) for args in zip(*all_leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_all(tree: ArrayLikeSpec, *, is_leaf: ArrayLikeSpec = None) -> bool:
    """Check whether all leaves in a PyTree evaluate to ``True``.

    Args:
        tree: The PyTree to check.
        is_leaf: Optional leaf predicate.

    Returns:
        bool: ``True`` if all leaves are truthy.
    """
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf)
    return all(bool(leaf) for leaf in leaves)


def tree_any(tree: ArrayLikeSpec, *, is_leaf: ArrayLikeSpec = None) -> bool:
    """Check whether any leaf in a PyTree evaluates to ``True``.

    Args:
        tree: The PyTree to check.
        is_leaf: Optional leaf predicate.

    Returns:
        bool: ``True`` if at least one leaf is truthy.
    """
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf)
    return any(bool(leaf) for leaf in leaves)


def _patch_pytreedef() -> None:
    """No-op compatibility hook retained for backward compatibility; ``PyTreeDef`` defines ``num_leaves`` and ``num_nodes`` natively."""
    return


__all__ = [
    "DictKey",
    "FlattenFunc",
    "FlattenedIndexKey",
    "GetAttrKey",
    "PyTreeDef",
    "SequenceKey",
    "UnflattenFunc",
    "_HasFieldsProtocol",
    "_TreeFlattenWithKeysProtocol",
    "_extract_key_child_pairs",
    "_flatten_with_path_recursive",
    "_is_namedtuple_instance",
    "_patch_pytreedef",
    "_pytree_registry",
    "_pytree_with_keys_registry",
    "_unflatten_from_iter",
    "register_pytree_node",
    "register_pytree_node_class",
    "register_pytree_with_keys",
    "register_pytree_with_keys_class",
    "tree_all",
    "tree_any",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_leaves",
    "tree_map",
    "tree_structure",
    "tree_unflatten",
]
