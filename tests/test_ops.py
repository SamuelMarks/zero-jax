"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.ops as mod


def test_segment_max() -> None:
    """Test segment_max."""
    with patch("zero_jax._compiler_proxy_ops.segment_max", create=True) as mock_op:
        mod.segment_max()
        mock_op.assert_called_once_with()


def test_segment_min() -> None:
    """Test segment_min."""
    with patch("zero_jax._compiler_proxy_ops.segment_min", create=True) as mock_op:
        mod.segment_min()
        mock_op.assert_called_once_with()


def test_segment_prod() -> None:
    """Test segment_prod."""
    with patch("zero_jax._compiler_proxy_ops.segment_prod", create=True) as mock_op:
        mod.segment_prod()
        mock_op.assert_called_once_with()


def test_segment_sum() -> None:
    """Test segment_sum."""
    with patch("zero_jax._compiler_proxy_ops.segment_sum", create=True) as mock_op:
        mod.segment_sum()
        mock_op.assert_called_once_with()


def test_Transpose_exists() -> None:
    """Test Transpose exists in ml_switcheroo_compiler."""
    import zero_jax._compiler_proxy_ops as ops

    assert hasattr(ops, "Transpose")
