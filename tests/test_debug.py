"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.debug as mod


def test_DebugEffect() -> None:
    """Test DebugEffect."""
    obj = mod.DebugEffect()
    assert obj is not None


def test_breakpoint() -> None:
    """Test breakpoint."""
    with patch("zero_jax._compiler_proxy_ops.breakpoint", create=True) as mock_op:
        mod.breakpoint()
        mock_op.assert_called_once_with()


def test_callback() -> None:
    """Test callback."""
    with patch("zero_jax._compiler_proxy_ops.callback", create=True) as mock_op:
        mod.callback()
        mock_op.assert_called_once_with()


def test_inspect_array_sharding() -> None:
    """Test inspect_array_sharding."""
    with patch(
        "zero_jax._compiler_proxy_ops.inspect_array_sharding", create=True
    ) as mock_op:
        mod.inspect_array_sharding()
        mock_op.assert_called_once_with()


def test_print() -> None:
    """Test print."""
    with patch("zero_jax._compiler_proxy_ops.print", create=True) as mock_op:
        mod.print()
        mock_op.assert_called_once_with()


def test_visualize_array_sharding() -> None:
    """Test visualize_array_sharding."""
    with patch(
        "zero_jax._compiler_proxy_ops.visualize_array_sharding", create=True
    ) as mock_op:
        mod.visualize_array_sharding()
        mock_op.assert_called_once_with()


def test_visualize_sharding() -> None:
    """Test visualize_sharding."""
    with patch(
        "zero_jax._compiler_proxy_ops.visualize_sharding", create=True
    ) as mock_op:
        mod.visualize_sharding()
        mock_op.assert_called_once_with()
