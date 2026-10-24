"""Tests for zero_jax.tree_util."""

from __future__ import annotations

import dataclasses

import pytest

import zero_jax
import zero_jax.tree_util as mod
from zero_jax._types import ArrayLikeSpec


def test_keys_and_keystr() -> None:
    """Verify DictKey, SequenceKey, GetAttrKey, FlattenedIndexKey, and keystr."""
    dk = mod.DictKey("w")
    sk = mod.SequenceKey(0)
    gk = mod.GetAttrKey("weight")
    fk = mod.FlattenedIndexKey(2)

    assert dk == mod.DictKey("w")
    assert sk == mod.SequenceKey(0)
    assert gk == mod.GetAttrKey("weight")
    assert fk == mod.FlattenedIndexKey(2)
    assert len({dk, sk, gk, fk}) == 4

    path = (dk, sk, gk, fk)
    assert mod.keystr(path) == "['w'][0].weight[<flat index 2>]"
    assert mod.keystr(path, simple=True) == "w/0/weight/2"
    assert mod.keystr((*path, "raw"), simple=True, separator=".") == "w.0.weight.2.raw"
    assert mod.keystr("scalar_key") == "scalar_key"


def test_Partial_pytree() -> None:
    """Verify Partial callable and PyTree flattening."""
    p = mod.Partial(lambda a, b, c=1: a + b + c, 10, c=5)
    assert p(20) == 35
    leaves, treedef = mod.tree_flatten(p)
    assert leaves == [10, 5]
    p2 = mod.tree_unflatten(treedef, [100, 50])
    assert p2(20) == 170
    assert p == mod.Partial(p.func, 10, c=5)


def test_tree_flatten_with_path_and_map() -> None:
    """Verify tree_flatten_with_path, tree_leaves_with_path, and tree_map_with_path."""
    tree = {"a": [10, 20], "b": (30,)}
    path_leaves, _treedef = mod.tree_flatten_with_path(tree)
    assert path_leaves == [
        ((mod.DictKey("a"), mod.SequenceKey(0)), 10),
        ((mod.DictKey("a"), mod.SequenceKey(1)), 20),
        ((mod.DictKey("b"), mod.SequenceKey(0)), 30),
    ]
    assert mod.tree_leaves_with_path(tree) == path_leaves

    mapped = mod.tree_map_with_path(lambda path, x: x + len(path), tree)
    assert mapped == {"a": [12, 22], "b": (32,)}

    mapped_two = mod.tree_map_with_path(lambda path, x, y: x + y, tree, tree)
    assert mapped_two == {"a": [20, 40], "b": (60,)}

    # Test is_leaf predicate
    leaf_paths, _ = mod.tree_flatten_with_path(tree, is_leaf=lambda node: isinstance(node, list))
    assert leaf_paths == [
        ((mod.DictKey("a"),), [10, 20]),
        ((mod.DictKey("b"), mod.SequenceKey(0)), 30),
    ]


def test_register_dataclass_and_static() -> None:
    """Verify register_dataclass, register_static, and unified zero_jax.tree routing."""

    @dataclasses.dataclass
    class Layer:
        """Sample dataclass for PyTree registration."""

        weight: int
        bias: int
        name: str = "linear"

    mod.register_dataclass(Layer, data_fields=["weight", "bias"], meta_fields=["name"])
    layer = Layer(weight=3, bias=4, name="dense")

    leaves, treedef = zero_jax.tree.flatten(layer)
    assert leaves == [3, 4]
    reconstructed = zero_jax.tree.unflatten(treedef, [30, 40])
    assert reconstructed == Layer(weight=30, bias=40, name="dense")

    path_leaves = mod.tree_leaves_with_path(layer)
    assert path_leaves == [
        ((mod.GetAttrKey("weight"),), 3),
        ((mod.GetAttrKey("bias"),), 4),
    ]

    @mod.register_static
    class StaticConfig:
        """Static node with zero leaves."""

        def __init__(self, lr: float) -> None:
            """Initialize StaticConfig."""
            self.lr = lr

    cfg = StaticConfig(0.01)
    cfg_leaves, cfg_def = mod.tree_flatten(cfg)
    assert cfg_leaves == []
    assert mod.tree_unflatten(cfg_def, []) is cfg

    # Non-type pass-through safety
    non_type_arg: ArrayLikeSpec = 123
    assert mod.register_dataclass(non_type_arg) == 123
    assert mod.register_static(non_type_arg) == 123


def test_tree_reduce_transpose_and_treedef_helpers() -> None:
    """Verify tree_reduce, tree_transpose, all_leaves, build_tree, and treedef helpers."""
    tree = {"a": [1, 2], "b": [3, 4]}
    assert mod.tree_reduce(lambda a, b: a + b, tree) == 10
    assert mod.tree_reduce(lambda a, b: a + b, tree, 5) == 15
    assert zero_jax.tree.reduce(lambda a, b: a + b, tree, 0) == 10
    assert zero_jax.tree.all({"a": True, "b": True}) is True
    assert zero_jax.tree.leaves(tree) == [1, 2, 3, 4]
    assert zero_jax.tree.map(lambda x: x + 1, {"k": 1}) == {"k": 2}

    outer_def = zero_jax.tree.structure({"a": 0, "b": 0})
    inner_def = zero_jax.tree.structure([0, 0])
    transposed = zero_jax.tree.transpose(outer_def, inner_def, tree)
    assert transposed == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    with pytest.raises(ValueError, match="Mismatch in leaves"):
        mod.tree_transpose(outer_def, inner_def, {"a": [1], "b": [2]})

    assert mod.all_leaves([1, 2, 3]) is True
    assert mod.all_leaves([1, [2, 3]]) is False

    built = mod.build_tree(outer_def, [10, 20])
    assert built == {"a": 10, "b": 20}

    children = mod.treedef_children(outer_def)
    assert len(children) == 2
    assert mod.treedef_is_leaf(children[0]) is True
    assert mod.treedef_is_leaf(outer_def) is False
    tup_def = mod.treedef_tuple(children)
    assert tup_def.num_leaves == 2


def test_custom_with_keys_and_plain_custom_nodes() -> None:
    """Verify custom nodes registered with register_pytree_node and register_pytree_with_keys_class."""

    @mod.register_pytree_node_class
    class Box:
        """Plain custom PyTree node."""

        def __init__(self, val: ArrayLikeSpec) -> None:
            """Initialize Box."""
            self.val = val

        def tree_flatten(self) -> tuple[tuple[ArrayLikeSpec], None]:
            """Flatten Box."""
            return (self.val,), None

        @classmethod
        def tree_unflatten(cls, aux: None, children: list[ArrayLikeSpec]) -> Box:
            """Unflatten Box."""
            return cls(children[0])

    b = Box(7)
    assert mod.tree_leaves_with_path(b) == [((mod.FlattenedIndexKey(0),), 7)]

    @mod.register_pytree_with_keys_class
    class KeyedBox:
        """Keyed custom PyTree node."""

        def __init__(self, val: ArrayLikeSpec) -> None:
            """Initialize KeyedBox."""
            self.val = val

        def tree_flatten_with_keys(self) -> tuple[tuple[tuple[mod.GetAttrKey, ArrayLikeSpec]], None]:
            """Flatten KeyedBox with keys."""
            return ((mod.GetAttrKey("val"), self.val),), None

        @classmethod
        def tree_unflatten(cls, aux: None, children: list[ArrayLikeSpec]) -> KeyedBox:
            """Unflatten KeyedBox."""
            return cls(children[0])

    kb = KeyedBox(9)
    assert mod.tree_leaves(kb) == [9]
    assert mod.tree_leaves_with_path(kb) == [((mod.GetAttrKey("val"),), 9)]
