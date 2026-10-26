"""Tests for zero_jax.typing."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.typing as mod
from tests.conftest import assert_allclose


def test_ArrayLike() -> None:
    """Test ArrayLike."""
    assert_allclose(mod.ArrayLike([1.0, 2.0]), [1.0, 2.0])


def test_DTypeLike() -> None:
    """Test DTypeLike."""
    assert mod.DTypeLike("float32") == jnp.float32
