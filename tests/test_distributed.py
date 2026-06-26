"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.distributed as mod


def test_initialize() -> None:
    """Test initialize."""
    with patch("ml_switcheroo_compiler.ops.initialize") as mock_op:
        mod.initialize()
        mock_op.assert_called_once_with()


def test_shutdown() -> None:
    """Test shutdown."""
    with patch("ml_switcheroo_compiler.ops.shutdown") as mock_op:
        mod.shutdown()
        mock_op.assert_called_once_with()
