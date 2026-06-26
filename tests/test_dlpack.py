"""Tests for zero_jax module."""

import pytest
import zero_jax.dlpack as mod


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    with pytest.raises(NotImplementedError):
        mod.from_dlpack()


def test_to_dlpack() -> None:
    """Test to_dlpack."""
    with pytest.raises(NotImplementedError):
        mod.to_dlpack()
