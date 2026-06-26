"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed as mod


def test_State() -> None:
    """Test State."""
    obj = mod.State()
    assert obj is not None


def test_initialize() -> None:
    """Test initialize."""
    with pytest.raises(NotImplementedError):
        mod.initialize()


def test_shutdown() -> None:
    """Test shutdown."""
    with pytest.raises(NotImplementedError):
        mod.shutdown()
