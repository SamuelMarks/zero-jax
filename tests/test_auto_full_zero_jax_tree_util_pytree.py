"""Tests for zero_jax.tree_util.pytree."""

from __future__ import annotations

import zero_jax.tree_util.pytree as mod
from zero_jax._types import ArrayLikeSpec


def test_pytreedef_and_equality() -> None:
    """Verify PyTreeDef properties and equality."""
    mod._patch_pytreedef()
    t1 = {"a": [1, 2], "b": (3,)}
    leaves, td1 = mod.tree_flatten(t1)
    assert leaves == [1, 2, 3]
    assert td1.num_leaves == 3
    assert td1.num_nodes == 6
    assert td1 == mod.tree_structure(t1)
    assert (td1 == "not_a_treedef") is False

    # Empty container has 0 leaves
    _, empty_td = mod.tree_flatten([])
    assert empty_td.num_leaves == 0

    # Unflatten does not mutate input leaf list
    original_leaves = [10, 20, 30]
    rebuilt = mod.tree_unflatten(td1, original_leaves)
    assert rebuilt == {"a": [10, 20], "b": (30,)}
    assert original_leaves == [10, 20, 30]


def test_tree_all_any_and_is_leaf() -> None:
    """Verify tree_all, tree_any, and is_leaf support."""
    assert mod.tree_all({"a": True, "b": [1, 2]}) is True
    assert mod.tree_all({"a": True, "b": [1, 0]}) is False
    assert mod.tree_any({"a": False, "b": [0, 1]}) is True
    assert mod.tree_any({"a": False, "b": [0, 0]}) is False

    leaves = mod.tree_leaves({"a": [1, 2]}, is_leaf=lambda x: isinstance(x, list))
    assert leaves == [[1, 2]]


def test_register_pytree_with_keys_variants() -> None:
    """Verify register_pytree_with_keys with and without explicit flatten_func."""

    class PairA:
        """Custom pair with explicit flatten_func."""

        def __init__(self, x: ArrayLikeSpec, y: ArrayLikeSpec) -> None:
            """Initialize PairA."""
            self.x = x
            self.y = y

    mod.register_pytree_with_keys(
        PairA,
        lambda p: ([("x", p.x), ("y", p.y)], None),
        lambda _, c: PairA(c[0], c[1]),
        flatten_func=lambda p: ([p.x, p.y], None),
    )
    pa = PairA(1, 2)
    assert mod.tree_leaves(pa) == [1, 2]

    class PairB:
        """Custom pair returning non-tuple children in flatten_with_keys."""

        def __init__(self, x: ArrayLikeSpec) -> None:
            """Initialize PairB."""
            self.x = x

    mod.register_pytree_with_keys(
        PairB,
        lambda p: ([p.x], None),
        lambda _, c: PairB(c[0]),
    )
    pb = PairB(5)
    assert mod.tree_leaves(pb) == [5]
    assert mod.tree_unflatten(mod.tree_structure(pb), [50]).x == 50

    @mod.register_pytree_with_keys_class
    class FallbackClass:
        """Class using register_pytree_with_keys_class without tree_flatten_with_keys."""

        def __init__(self, v: ArrayLikeSpec) -> None:
            """Initialize FallbackClass."""
            self.v = v

        def tree_flatten(self) -> tuple[list[ArrayLikeSpec], None]:
            """Flatten FallbackClass."""
            return [self.v], None

        @classmethod
        def tree_unflatten(cls, aux: None, children: list[ArrayLikeSpec]) -> FallbackClass:
            """Unflatten FallbackClass."""
            return cls(children[0])

    fc = FallbackClass(11)
    assert mod.tree_leaves(fc) == [11]
    assert mod.tree_unflatten(mod.tree_structure(fc), [22]).v == 22
