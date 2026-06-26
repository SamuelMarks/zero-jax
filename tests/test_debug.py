"""Tests for zero_jax module."""

import pytest
import zero_jax.debug as mod


def test_DebugEffect() -> None:
    """Test DebugEffect."""
    obj = mod.DebugEffect()
    assert obj is not None


def test_breakpoint() -> None:
    """Test breakpoint."""
    with pytest.raises(NotImplementedError):
        mod.breakpoint()


def test_callback() -> None:
    """Test callback."""
    with pytest.raises(NotImplementedError):
        mod.callback()


def test_inspect_array_sharding() -> None:
    """Test inspect_array_sharding."""
    with pytest.raises(NotImplementedError):
        mod.inspect_array_sharding()


def test_print() -> None:
    """Test print."""
    with pytest.raises(NotImplementedError):
        mod.print()


def test_visualize_array_sharding() -> None:
    """Test visualize_array_sharding."""
    with pytest.raises(NotImplementedError):
        mod.visualize_array_sharding()


def test_visualize_sharding() -> None:
    """Test visualize_sharding."""
    with pytest.raises(NotImplementedError):
        mod.visualize_sharding()
