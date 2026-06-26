"""Tests for zero_jax module."""

import pytest
import zero_jax.profiler as mod


def test_StepTraceAnnotation() -> None:
    """Test StepTraceAnnotation."""
    obj = mod.StepTraceAnnotation()
    assert obj is not None


def test_annotate_function() -> None:
    """Test annotate_function."""
    with pytest.raises(NotImplementedError):
        mod.annotate_function()


def test_device_memory_profile() -> None:
    """Test device_memory_profile."""
    with pytest.raises(NotImplementedError):
        mod.device_memory_profile()


def test_save_device_memory_profile() -> None:
    """Test save_device_memory_profile."""
    with pytest.raises(NotImplementedError):
        mod.save_device_memory_profile()


def test_start_server() -> None:
    """Test start_server."""
    with pytest.raises(NotImplementedError):
        mod.start_server()


def test_start_trace() -> None:
    """Test start_trace."""
    with pytest.raises(NotImplementedError):
        mod.start_trace()


def test_stop_server() -> None:
    """Test stop_server."""
    with pytest.raises(NotImplementedError):
        mod.stop_server()


def test_stop_trace() -> None:
    """Test stop_trace."""
    with pytest.raises(NotImplementedError):
        mod.stop_trace()


def test_trace() -> None:
    """Test trace."""
    with pytest.raises(NotImplementedError):
        mod.trace()
