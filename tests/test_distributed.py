"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.distributed as mod


def test_initialize() -> None:
    """Test initialize."""
    with patch("zero_jax._compiler_proxy_ops.initialize", create=True) as mock_op:
        mod.initialize()
        mock_op.assert_called_once_with()


def test_shutdown() -> None:
    """Test shutdown."""
    with patch("zero_jax._compiler_proxy_ops.shutdown", create=True) as mock_op:
        mod.shutdown()
        mock_op.assert_called_once_with()
