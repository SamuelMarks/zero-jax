import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import zero_jax.tree_util as jtu


def test_tree_util_added():
    assert isinstance(jtu.DictKey(0), jtu.DictKey)
    assert isinstance(jtu.FlattenedIndexKey(0), jtu.FlattenedIndexKey)
    assert isinstance(jtu.GetAttrKey(0), jtu.GetAttrKey)
    assert isinstance(jtu.SequenceKey(0), jtu.SequenceKey)

    p = jtu.Partial(lambda x, y: x + y, 1)
    assert p(2) == 3

    assert jtu.all_leaves([1, 2])

    tree = {"a": 1, "b": 2}
    leaves, treedef = jtu.tree_flatten(tree)
    assert jtu.build_tree(treedef, leaves) == tree

    assert jtu.default_registry is not None
    assert isinstance(jtu.keystr([]), str)

    jtu.register_dataclass(None, None, None)
    jtu.register_pytree_node(None, None, None)

    @jtu.register_pytree_node_class
    class A:
        pass

    assert A is not None

    jtu.register_pytree_with_keys(None, None, None)

    @jtu.register_pytree_with_keys_class
    class B:
        pass

    assert B is not None

    @jtu.register_static
    class C:
        pass

    assert C is not None

    leaves_with_path, treedef2 = jtu.tree_flatten_with_path(tree)
    assert len(leaves_with_path) == 2
    assert len(jtu.tree_leaves_with_path(tree)) == 2

    mapped = jtu.tree_map_with_path(lambda p, x: x + 1, tree)
    assert mapped["a"] == 2

    assert jtu.tree_reduce(lambda x, y: x + y, tree) == 3

    tree2 = {"a": {"x": 1}, "b": {"x": 2}}
    _, outer_treedef = jtu.tree_flatten({"a": 0, "b": 0})
    _, inner_treedef = jtu.tree_flatten({"x": 0})
    assert jtu.tree_transpose(outer_treedef, inner_treedef, tree2) is not None

    leaf_treedef = jtu.tree_flatten(1)[1]
    assert jtu.treedef_children(leaf_treedef) == []
    assert not jtu.treedef_is_leaf(treedef)
    assert jtu.treedef_tuple([treedef]) == jtu.tree_flatten((tree,))[1]
