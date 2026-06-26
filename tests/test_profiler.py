"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.profiler as mod


def test_StepTraceAnnotation() -> None:
    """Test StepTraceAnnotation."""
    obj = mod.StepTraceAnnotation()
    assert obj is not None


def test_annotate_function() -> None:
    """Test annotate_function."""
    with patch("ml_switcheroo_compiler.ops.annotate_function") as mock_op:
        mod.annotate_function()
        mock_op.assert_called_once_with()


def test_device_memory_profile() -> None:
    """Test device_memory_profile."""
    with patch("ml_switcheroo_compiler.ops.device_memory_profile") as mock_op:
        mod.device_memory_profile()
        mock_op.assert_called_once_with()


def test_save_device_memory_profile() -> None:
    """Test save_device_memory_profile."""
    with patch("ml_switcheroo_compiler.ops.save_device_memory_profile") as mock_op:
        mod.save_device_memory_profile()
        mock_op.assert_called_once_with()


def test_start_server() -> None:
    """Test start_server."""
    with patch("ml_switcheroo_compiler.ops.start_server") as mock_op:
        mod.start_server()
        mock_op.assert_called_once_with()


def test_start_trace() -> None:
    """Test start_trace."""
    with patch("ml_switcheroo_compiler.ops.start_trace") as mock_op:
        mod.start_trace()
        mock_op.assert_called_once_with()


def test_stop_server() -> None:
    """Test stop_server."""
    with patch("ml_switcheroo_compiler.ops.stop_server") as mock_op:
        mod.stop_server()
        mock_op.assert_called_once_with()


def test_stop_trace() -> None:
    """Test stop_trace."""
    with patch("ml_switcheroo_compiler.ops.stop_trace") as mock_op:
        mod.stop_trace()
        mock_op.assert_called_once_with()


def test_trace() -> None:
    """Test trace."""
    with patch("ml_switcheroo_compiler.ops.trace") as mock_op:
        mod.trace()
        mock_op.assert_called_once_with()
