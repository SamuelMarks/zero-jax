"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.dlpack as mod


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    with patch("ml_switcheroo_compiler.ops.from_dlpack") as mock_op:
        mod.from_dlpack()
        mock_op.assert_called_once_with()


def test_to_dlpack() -> None:
    """Test to_dlpack."""
    with patch("ml_switcheroo_compiler.ops.to_dlpack") as mock_op:
        mod.to_dlpack()
        mock_op.assert_called_once_with()
