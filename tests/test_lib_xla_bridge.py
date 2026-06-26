"""Tests for zero_jax module."""

import pytest
import zero_jax.lib.xla_bridge as mod


def test_default_backend() -> None:
    """Test default_backend."""
    with pytest.raises(NotImplementedError):
        mod.default_backend()


def test_get_backend() -> None:
    """Test get_backend."""
    with pytest.raises(NotImplementedError):
        mod.get_backend()


def test_get_compile_options() -> None:
    """Test get_compile_options."""
    with pytest.raises(NotImplementedError):
        mod.get_compile_options()
