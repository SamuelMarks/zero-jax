"""Tests for zero_jax module."""

import pytest
import zero_jax.image as mod


def test_ResizeMethod() -> None:
    """Test ResizeMethod."""
    obj = mod.ResizeMethod()
    assert obj is not None


def test_resize() -> None:
    """Test resize."""
    with pytest.raises(NotImplementedError):
        mod.resize()


def test_scale_and_translate() -> None:
    """Test scale_and_translate."""
    with pytest.raises(NotImplementedError):
        mod.scale_and_translate()
