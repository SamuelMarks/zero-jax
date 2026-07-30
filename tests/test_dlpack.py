"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.dlpack as mod


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    with patch("zero_jax._compiler_proxy_ops.from_dlpack", create=True) as mock_op:
        mod.from_dlpack()
        mock_op.assert_called_once_with()


def test_to_dlpack() -> None:
    """Test to_dlpack."""
    with patch("zero_jax._compiler_proxy_ops.to_dlpack", create=True) as mock_op:
        mod.to_dlpack()
        mock_op.assert_called_once_with()
