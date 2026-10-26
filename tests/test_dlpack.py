"""Tests for zero_jax.dlpack."""

from __future__ import annotations

import zero_jax.dlpack as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    arr = jnp.array([1.0, 2.0])
    assert_allclose(mod.from_dlpack(arr), [1.0, 2.0])


def test_to_dlpack() -> None:
    """Test to_dlpack."""
    arr = jnp.array([3.0, 4.0])
    assert_allclose(mod.to_dlpack(arr), [3.0, 4.0])
