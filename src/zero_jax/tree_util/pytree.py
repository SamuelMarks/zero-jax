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
    if isinstance(tree, tuple):
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
    if treedef.node_type is type(None):
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
