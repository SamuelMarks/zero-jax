"""Tests for tests/test_lax_reduce.py."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.lax import primitives as lax
from zero_jax.numpy import array


def add(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``add``.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
    """
    return x + y


def mul(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``mul``.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
    """
    return x * y


def max(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``max``.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
    """
    return x if x > y else y


def min(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``min``.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
    """
    return x if x < y else y


def test_lax_reduce_others() -> None:
    """Execute ``test_lax_reduce_others``."""
    assert add(1, 2) == 3
    assert mul(2, 3) == 6
    assert max(2, 3) == 3
    assert min(2, 3) == 2

    x = array([1, 2, 3])

    res1 = lax.reduce(x, 1, mul, (0,))
    lax.reduce(x, 0, max, (0,))
    lax.reduce(x, 10, min, (0,))
    lax.reduce(x, 0, lambda a, b: a, (0,))

    assert res1 == 6
