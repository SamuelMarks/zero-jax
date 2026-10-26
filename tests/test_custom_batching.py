"""Tests for zero_jax.custom_batching."""

from __future__ import annotations

import zero_jax.custom_batching as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_custom_vmap() -> None:
    """Test custom_vmap."""
    obj = mod.custom_vmap()
    assert obj() is None
    fn = mod.custom_vmap(lambda x: x + 1)
    rule = lambda axis_size, in_batched, x: (x + 1, True)
    assert fn.def_vmap(rule) is rule
    assert fn(3) == 4


def test_sequential_vmap() -> None:
    """Test sequential_vmap."""
    vmapped = mod.sequential_vmap(lambda x: x * 2.0)
    assert_allclose(vmapped(jnp.array([1.0, 2.0])), [2.0, 4.0])
