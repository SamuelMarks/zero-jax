"""Tests for zero_jax module."""

import pytest
import zero_jax.distributed as mod


def test_initialize() -> None:
    """Test initialize."""
    with pytest.raises(NotImplementedError):
        mod.initialize()


def test_shutdown() -> None:
    """Test shutdown."""
    with pytest.raises(NotImplementedError):
        mod.shutdown()
