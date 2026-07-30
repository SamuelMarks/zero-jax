"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.profiler as mod


def test_StepTraceAnnotation() -> None:
    """Test StepTraceAnnotation."""
    obj = mod.StepTraceAnnotation()
    assert obj is not None


def test_TraceAnnotation() -> None:
    """Test TraceAnnotation."""
    obj = mod.TraceAnnotation()
    assert obj is not None


def test_annotate_function() -> None:
    """Test annotate_function."""
    with patch(
        "zero_jax._compiler_proxy_ops.annotate_function", create=True
    ) as mock_op:
        mod.annotate_function()
        mock_op.assert_called_once_with()


def test_device_memory_profile() -> None:
    """Test device_memory_profile."""
    with patch(
        "zero_jax._compiler_proxy_ops.device_memory_profile", create=True
    ) as mock_op:
        mod.device_memory_profile()
        mock_op.assert_called_once_with()


def test_save_device_memory_profile() -> None:
    """Test save_device_memory_profile."""
    with patch(
        "zero_jax._compiler_proxy_ops.save_device_memory_profile", create=True
    ) as mock_op:
        mod.save_device_memory_profile()
        mock_op.assert_called_once_with()


def test_start_server() -> None:
    """Test start_server."""
    with patch("zero_jax._compiler_proxy_ops.start_server", create=True) as mock_op:
        mod.start_server()
        mock_op.assert_called_once_with()


def test_start_trace() -> None:
    """Test start_trace."""
    with patch("zero_jax._compiler_proxy_ops.start_trace", create=True) as mock_op:
        mod.start_trace()
        mock_op.assert_called_once_with()


def test_stop_server() -> None:
    """Test stop_server."""
    with patch("zero_jax._compiler_proxy_ops.stop_server", create=True) as mock_op:
        mod.stop_server()
        mock_op.assert_called_once_with()


def test_stop_trace() -> None:
    """Test stop_trace."""
    with patch("zero_jax._compiler_proxy_ops.stop_trace", create=True) as mock_op:
        mod.stop_trace()
        mock_op.assert_called_once_with()


def test_trace() -> None:
    """Test trace."""
    with patch("zero_jax._compiler_proxy_ops.trace", create=True) as mock_op:
        mod.trace()
        mock_op.assert_called_once_with()
