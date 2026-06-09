"""PyTree manipulation utilities."""

from typing import Any, Tuple, List


class PyTreeDef:
    """Represents the structure of a PyTree."""

    def __init__(
        self, node_type: type, children_defs: List["PyTreeDef"], metadata: Any = None
    ):
        self.node_type = node_type
        self.children_defs = children_defs
        self.metadata = metadata


def tree_flatten(tree: Any) -> Tuple[List[Any], PyTreeDef]:
    """Flattens a pytree into a list of leaves and a treedef.

    Args:
        tree: The PyTree to flatten.

    Returns:
        A tuple of (leaves, treedef).
    """
    if isinstance(tree, tuple):
        leaves = []
        child_defs = []
        for child in tree:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            child_defs.append(child_def)
        return leaves, PyTreeDef(tuple, child_defs, metadata=len(tree))

    elif isinstance(tree, list):
        leaves = []
        child_defs = []
        for child in tree:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            child_defs.append(child_def)
        return leaves, PyTreeDef(list, child_defs, metadata=len(tree))

    elif isinstance(tree, dict):
        leaves = []
        child_defs = []
        keys = sorted(tree.keys())
        for k in keys:
            child_leaves, child_def = tree_flatten(tree[k])
            leaves.extend(child_leaves)
            child_defs.append(child_def)
        return leaves, PyTreeDef(dict, child_defs, metadata=keys)

    else:
        return [tree], PyTreeDef(type(tree), [], metadata=None)


def tree_unflatten(treedef: PyTreeDef, leaves: List[Any]) -> Any:
    """Reconstructs a PyTree from leaves and a treedef.

    Args:
        treedef: The structure definition.
        leaves: The flattened leaves.

    Returns:
        The unflattened PyTree.
    """
    leaves_copy = list(leaves)

    def _unflatten(tdef: PyTreeDef) -> Any:
        if tdef.node_type is tuple:
            children = [_unflatten(c) for c in tdef.children_defs]
            return tuple(children)
        elif tdef.node_type is list:
            children = [_unflatten(c) for c in tdef.children_defs]
            return children
        elif tdef.node_type is dict:
            children = [_unflatten(c) for c in tdef.children_defs]
            return {k: v for k, v in zip(tdef.metadata, children)}
        else:
            return leaves_copy.pop(0)

    return _unflatten(treedef)
