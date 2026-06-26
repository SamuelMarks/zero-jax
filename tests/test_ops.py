"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.ops as mod


def test_segment_max() -> None:
    """Test segment_max."""
    with patch("ml_switcheroo_compiler.ops.segment_max") as mock_op:
        mod.segment_max()
        mock_op.assert_called_once_with()


def test_segment_min() -> None:
    """Test segment_min."""
    with patch("ml_switcheroo_compiler.ops.segment_min") as mock_op:
        mod.segment_min()
        mock_op.assert_called_once_with()


def test_segment_prod() -> None:
    """Test segment_prod."""
    with patch("ml_switcheroo_compiler.ops.segment_prod") as mock_op:
        mod.segment_prod()
        mock_op.assert_called_once_with()


def test_segment_sum() -> None:
    """Test segment_sum."""
    with patch("ml_switcheroo_compiler.ops.segment_sum") as mock_op:
        mod.segment_sum()
        mock_op.assert_called_once_with()


def test_Transpose_exists() -> None:
    """Test Transpose exists in ml_switcheroo_compiler."""
    import ml_switcheroo_compiler.ops as ops

    assert hasattr(ops, "Transpose")
