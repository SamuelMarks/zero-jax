"""Tests for tests/test_transformations.py."""

from __future__ import annotations

from zero_jax import disable_jit, grad, jit, pmap, value_and_grad, vmap
from zero_jax._types import ArrayLikeSpec


def test_jit() -> None:
    """Execute ``test_jit``."""

    @jit
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``."""
        return x + 1

    assert f(1) == 2


def test_grad() -> None:
    """Execute ``test_grad``."""

    @grad
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``."""
        return x * x

    assert f(2) == 4


def test_value_and_grad() -> None:
    """Execute ``test_value_and_grad``."""

    @value_and_grad
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``."""
        return x * 2

    val, g = f(2)
    assert val == 4
    assert g == 2


def test_vmap() -> None:
    """Execute ``test_vmap``."""

    @vmap
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``."""
        return x

    assert f(5) == 5


def test_disable_jit() -> None:
    """Execute ``test_disable_jit``."""
    with disable_jit():
        pass


def test_pmap() -> None:
    """Execute ``test_pmap``."""

    @pmap
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f``."""
        return x

    assert f(5) == 5
