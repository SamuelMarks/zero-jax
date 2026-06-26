"""Tests for zero_jax module."""

import pytest
import zero_jax.custom_batching as mod


def test_custom_vmap() -> None:
    """Test custom_vmap."""
    obj = mod.custom_vmap()
    assert obj is not None


def test_sequential_vmap() -> None:
    """Test sequential_vmap."""
    with pytest.raises(NotImplementedError):
        mod.sequential_vmap()
