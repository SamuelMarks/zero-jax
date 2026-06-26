"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.image as mod


def test_ResizeMethod() -> None:
    """Test ResizeMethod."""
    obj = mod.ResizeMethod()
    assert obj is not None


def test_resize() -> None:
    """Test resize."""
    with patch("ml_switcheroo_compiler.ops.resize") as mock_op:
        mod.resize()
        mock_op.assert_called_once_with()


def test_scale_and_translate() -> None:
    """Test scale_and_translate."""
    with patch("ml_switcheroo_compiler.ops.scale_and_translate") as mock_op:
        mod.scale_and_translate()
        mock_op.assert_called_once_with()
