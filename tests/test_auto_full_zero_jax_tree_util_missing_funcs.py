"""Tests for zero_jax.tree_util.missing_funcs."""

from __future__ import annotations

import zero_jax.tree_util.missing_funcs as mod
from zero_jax._types import ArrayLikeSpec


def test_missing_funcs_non_paired_key_children() -> None:
    """Verify _flatten_with_path_recursive handles custom nodes whose flatten_with_keys yields plain values."""

    class RawKeyNode:
        """Custom node returning plain children from flatten_with_keys."""

        def __init__(self, val: ArrayLikeSpec) -> None:
            """Initialize RawKeyNode."""
            self.val = val

    mod.register_pytree_with_keys(
        RawKeyNode,
        lambda n: ([n.val], None),
        lambda _, c: RawKeyNode(c[0]),
    )
    node = RawKeyNode(42)
    path_leaves, _ = mod.tree_flatten_with_path(node)
    assert path_leaves == [((mod.FlattenedIndexKey(0),), 42)]
