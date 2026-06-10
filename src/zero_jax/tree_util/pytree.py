"""PyTree manipulation utilities."""

from typing import Any

from typing import Tuple, List


class PyTreeDef:
    """PyTreeDef class."""

    @property
    def num_leaves(self) -> Any:
        """num_leaves function."""
        if not self.children_defs:
            return 1
        return sum(c.num_leaves for c in self.children_defs)

    @property
    def num_nodes(self) -> Any:
        """num_nodes function."""
        return 1 + sum(c.num_nodes for c in self.children_defs)

    """Represents the structure of a PyTree."""

    def __init__(
        self, node_type: type, children_defs: List["PyTreeDef"], metadata: Any = None
    ) -> None:
        """Initialize."""
        self.node_type = node_type
        self.children_defs = children_defs
        self.metadata = metadata


def tree_flatten(tree: Any) -> Tuple[List[Any], PyTreeDef]:
    """tree_flatten function."""
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
    """tree_unflatten function."""
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


def tree_leaves(tree: Any) -> List[Any]:
    """tree_leaves function."""
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: Any) -> PyTreeDef:
    """tree_structure function."""
    _, treedef = tree_flatten(tree)
    return treedef


def tree_map(f: Any, tree: Any, *rest: Any) -> Any:
    """tree_map function."""
    leaves, treedef = tree_flatten(tree)
    all_leaves = [leaves]
    for r in rest:
        r_leaves, _ = tree_flatten(r)
        all_leaves.append(r_leaves)
    mapped_leaves = [f(*args) for args in zip(*all_leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_all(tree: Any) -> bool:
    """tree_all function."""
    # JAX tree_all takes a single tree and evaluates truthiness of leaves
    # Wait, JAX tree_all actually takes a function `tree_all(f, tree)`.
    pass


def tree_any(tree: Any) -> bool:
    """tree_any function."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if leaf:
            return True
    return False


# Also let PyTreeDef have num_nodes and num_leaves for parity
def _patch_pytreedef() -> Any:
    """_patch_pytreedef function."""
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
