"""Tests for zero_jax module."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.lib.xla_bridge as mod


def test_default_backend() -> None:
    """Test default_backend."""
    with patch("zero_jax._compiler_proxy_ops.default_backend", create=True) as mock_op:
        mod.default_backend()
        mock_op.assert_called_once_with()


def test_get_backend() -> None:
    """Test get_backend."""
    with patch("zero_jax._compiler_proxy_ops.get_backend", create=True) as mock_op:
        mod.get_backend()
        mock_op.assert_called_once_with()


def test_get_compile_options() -> None:
    """Test get_compile_options."""
    with patch("zero_jax._compiler_proxy_ops.get_compile_options", create=True) as mock_op:
        mod.get_compile_options()
        mock_op.assert_called_once_with()
