"""PyTree manipulation utilities."""

from __future__ import annotations

from typing import Any, List, Tuple

_pytree_registry = {}


def register_pytree_node(
    nodetype: type, flatten_func: Any, unflatten_func: Any
) -> None:
    """Registers a new type with the PyTree registry."""
    _pytree_registry[nodetype] = (flatten_func, unflatten_func)


def register_pytree_node_class(cls: type) -> type:
    """Registers a class with the PyTree registry using its methods."""
    register_pytree_node(
        cls,
        lambda obj: obj.tree_flatten(),
        lambda aux, children: cls.tree_unflatten(aux, children),
    )
    return cls


def register_pytree_with_keys(
    nodetype: type, flatten_func: Any, unflatten_func: Any
) -> None:
    """Registers a new type with the PyTree registry (ignores keys for now)."""
    _pytree_registry[nodetype] = (flatten_func, unflatten_func)


def register_pytree_with_keys_class(cls: type) -> type:
    """Registers a class with the PyTree registry using its methods."""
    return register_pytree_node_class(cls)


class PyTreeDef:
    """Represents the structure of a PyTree.

    Attributes:
        node_type: The type of the node.
        children_defs: The definitions of the children nodes.
        metadata: Additional metadata for the node.
    """

    @property
    def num_leaves(self) -> Any:
        """Returns the number of leaves in the PyTree structure.

        Returns:
            The total number of leaf nodes.
        """
        if not self.children_defs:
            return 1
        return sum(c.num_leaves for c in self.children_defs)

    @property
    def num_nodes(self) -> Any:
        """Returns the number of nodes in the PyTree structure.

        Returns:
            The total number of all nodes (internal and leaves).
        """
        return 1 + sum(c.num_nodes for c in self.children_defs)

    def __eq__(self, other):
        """JAX API implementation for __eq__.

        Args:
            self: Argument self.
            other: Argument other.

        Returns:
            Any: The result.
        """
        if not isinstance(other, PyTreeDef):
            return False
        return (
            self.node_type == other.node_type
            and self.metadata == other.metadata
            and self.children_defs == other.children_defs
        )

    def __init__(
        self, node_type: type, children_defs: List[PyTreeDef], metadata: Any = None
    ) -> None:
        """Initializes a PyTreeDef.

        Args:
            node_type: The type of the node.
            children_defs: A list of children node definitions.
            metadata: Any metadata associated with the node.
        """
        self.node_type = node_type
        self.children_defs = children_defs
        self.metadata = metadata


def tree_flatten(tree: Any) -> Tuple[List[Any], PyTreeDef]:
    """Flattens a PyTree into a list of leaves and a PyTreeDef structure.

    Args:
        tree: The PyTree to flatten.

    Returns:
        A tuple containing a list of leaves and the PyTree structure definition.
    """
    node_type = type(tree)
    if node_type in _pytree_registry:
        flatten_func, _ = _pytree_registry[node_type]  # pragma: no cover
        res = flatten_func(tree)  # pragma: no cover
        if len(res) == 2:  # pragma: no cover
            children, aux = res  # pragma: no cover
        else:
            children, aux = res[0], None  # pragma: no cover
        leaves = []  # pragma: no cover
        children_defs = []  # pragma: no cover
        for child in children:  # pragma: no cover
            child_leaves, child_def = tree_flatten(child)  # pragma: no cover
            leaves.extend(child_leaves)  # pragma: no cover
            children_defs.append(child_def)  # pragma: no cover
        return leaves, PyTreeDef(
            node_type, children_defs, metadata=aux
        )  # pragma: no cover
    elif isinstance(tree, tuple):
        leaves = []
        children_defs = []
        for child in tree:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, PyTreeDef(tuple, children_defs)
    elif isinstance(tree, list):
        leaves = []
        children_defs = []
        for child in tree:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, PyTreeDef(list, children_defs)
    elif isinstance(tree, dict):
        leaves = []
        children_defs = []
        keys = sorted(tree.keys())
        for k in keys:
            child_leaves, child_def = tree_flatten(tree[k])
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, PyTreeDef(dict, children_defs, metadata=keys)
    else:
        return [tree], PyTreeDef(type(None), [])


def tree_unflatten(treedef: PyTreeDef, leaves: List[Any]) -> Any:
    """Reconstructs a PyTree from its leaves and its PyTreeDef structure.

    Args:
        treedef: The PyTree structure definition.
        leaves: The list of leaves to insert into the PyTree.

    Returns:
        The reconstructed PyTree.
    """
    if treedef.node_type in _pytree_registry:
        _, unflatten_func = _pytree_registry[treedef.node_type]  # pragma: no cover
        children = []  # pragma: no cover
        for child_def in treedef.children_defs:  # pragma: no cover
            children.append(tree_unflatten(child_def, leaves))  # pragma: no cover
        return unflatten_func(treedef.metadata, children)  # pragma: no cover
    elif treedef.node_type is type(None):
        return leaves.pop(0)
    elif treedef.node_type is tuple:
        children = []
        for child_def in treedef.children_defs:
            children.append(tree_unflatten(child_def, leaves))
        return tuple(children)
    elif treedef.node_type is list:
        children = []
        for child_def in treedef.children_defs:
            children.append(tree_unflatten(child_def, leaves))
        return list(children)
    elif treedef.node_type is dict:
        children = {}
        keys = treedef.metadata
        for k, child_def in zip(keys, treedef.children_defs):
            children[k] = tree_unflatten(child_def, leaves)
        return children
    else:
        # Fallback
        return leaves.pop(0)


def tree_leaves(tree: Any) -> List[Any]:
    """Extracts all the leaves from a PyTree.

    Args:
        tree: The PyTree to extract leaves from.

    Returns:
        A list of the leaves of the PyTree.
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: Any) -> PyTreeDef:
    """Extracts the structure of a PyTree.

    Args:
        tree: The PyTree to analyze.

    Returns:
        The PyTreeDef structure of the PyTree.
    """
    _, treedef = tree_flatten(tree)
    return treedef


def tree_map(f: Any, tree: Any, *rest: Any) -> Any:
    """Maps a function over the leaves of a PyTree.

    Args:
        f: The function to map.
        tree: The primary PyTree.
        *rest: Additional PyTrees to zip with the primary tree.

    Returns:
        A new PyTree with the function applied to its leaves.
    """
    leaves, treedef = tree_flatten(tree)
    all_leaves = [leaves]
    for r in rest:
        r_leaves, _ = tree_flatten(r)
        all_leaves.append(r_leaves)
    mapped_leaves = [f(*args) for args in zip(*all_leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_all(tree: Any) -> bool:
    """Checks if all leaves in a PyTree evaluate to True.

    Args:
        tree: The PyTree to check.

    Returns:
        True if all leaves are truthy, False otherwise.
    """
    leaves, _ = tree_flatten(tree)
    return all(leaves)


def tree_any(tree: Any) -> bool:
    """Checks if any leaf in a PyTree evaluates to True.

    Args:
        tree: The PyTree to check.

    Returns:
        True if at least one leaf is truthy, False otherwise.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if leaf:
            return True
    return False


# Also let PyTreeDef have num_nodes and num_leaves for parity
def _patch_pytreedef() -> Any:
    """Patches PyTreeDef with num_leaves and num_nodes properties.

    Returns:
        None
    """
    PyTreeDef.num_leaves = property(
        lambda self: (
            1
            if not self.children_defs
            else sum(c.num_leaves for c in self.children_defs)
        )
    )
    PyTreeDef.num_nodes = property(
        lambda self: 1 + sum(c.num_nodes for c in self.children_defs)
    )
