"""Tests for zero_jax.profiler."""

from __future__ import annotations

import zero_jax.profiler as mod


def test_StepTraceAnnotation() -> None:
    """Test StepTraceAnnotation."""
    with mod.StepTraceAnnotation("train_step", step_num=3) as obj:
        assert obj.name == "train_step"
        assert obj.step_num == 3


def test_TraceAnnotation() -> None:
    """Test TraceAnnotation."""
    with mod.TraceAnnotation("forward") as obj:
        assert obj.name == "forward"


def test_annotate_function() -> None:
    """Test annotate_function."""
    fn = mod.annotate_function(lambda x: x + 2, name="add2")
    assert fn(5) == 7


def test_device_memory_profile() -> None:
    """Test device_memory_profile."""
    assert mod.device_memory_profile("cpu") == b""


def test_save_device_memory_profile() -> None:
    """Test save_device_memory_profile."""
    mod.save_device_memory_profile("/tmp/mem.prof", backend="cpu")


def test_start_server() -> None:
    """Test start_server."""
    mod.start_server(9001)


def test_start_trace() -> None:
    """Test start_trace."""
    mod.start_trace("/tmp/trace")


def test_stop_server() -> None:
    """Test stop_server."""
    mod.stop_server()


def test_stop_trace() -> None:
    """Test stop_trace."""
    mod.stop_trace()


def test_trace() -> None:
    """Test trace."""
    with mod.trace("/tmp/trace") as ctx:
        assert ctx is None
