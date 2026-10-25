"""Tests for zero_jax.tree_util.missing_funcs."""

from __future__ import annotations

import zero_jax.tree_util.missing_funcs as mod
from zero_jax._types import ArrayLikeSpec


def test_missing_funcs_non_paired_key_children() -> None:
    """Verify _flatten_with_path_recursive handles custom nodes whose flatten_with_keys yields plain values."""

    class RawKeyNode:
        """Custom node returning plain children from flatten_with_keys."""

        def __init__(self, val: ArrayLikeSpec) -> None:
            """Initialize RawKeyNode.

            Args:
                val: Parameter ``val``.
            """
            self.val = val

    mod.register_pytree_with_keys(
        RawKeyNode,
        lambda n: ([n.val], None),
        lambda _, c: RawKeyNode(c[0]),
    )
    node = RawKeyNode(42)
    path_leaves, _ = mod.tree_flatten_with_path(node)
    assert path_leaves == [((mod.FlattenedIndexKey(0),), 42)]

    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    pt = Point(10, 20)
    pt_leaves, pt_def = mod.tree_flatten(pt)
    assert pt_leaves == [10, 20]
    assert mod.tree_unflatten(pt_def, [30, 40]) == Point(30, 40)
    pt_path_leaves, _ = mod.tree_flatten_with_path(pt)
    assert pt_path_leaves == [((mod.GetAttrKey("x"),), 10), ((mod.GetAttrKey("y"),), 20)]
