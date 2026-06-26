"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.lib.xla_bridge as mod


def test_default_backend() -> None:
    """Test default_backend."""
    with patch("ml_switcheroo_compiler.ops.default_backend") as mock_op:
        mod.default_backend()
        mock_op.assert_called_once_with()


def test_get_backend() -> None:
    """Test get_backend."""
    with patch("ml_switcheroo_compiler.ops.get_backend") as mock_op:
        mod.get_backend()
        mock_op.assert_called_once_with()


def test_get_compile_options() -> None:
    """Test get_compile_options."""
    with patch("ml_switcheroo_compiler.ops.get_compile_options") as mock_op:
        mod.get_compile_options()
        mock_op.assert_called_once_with()
