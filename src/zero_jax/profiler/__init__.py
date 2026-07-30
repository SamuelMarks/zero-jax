"""Frontend API routing for jax.profiler."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class StepTraceAnnotation:
    """Context manager that generates a step trace event in the profiler."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class TraceAnnotation:
    """Context manager that generates a trace event in the profiler."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def annotate_function(*args: Any, **kwargs: Any) -> Any:
    """Decorator that generates a trace event for the execution of a function."""
    return _ops.annotate_function(*args, **kwargs)


def device_memory_profile(*args: Any, **kwargs: Any) -> Any:
    """Captures a JAX device memory profile as ``pprof``-format protocol buffer."""
    return _ops.device_memory_profile(*args, **kwargs)


def save_device_memory_profile(*args: Any, **kwargs: Any) -> Any:
    """Collects a device memory profile and writes it to a file."""
    return _ops.save_device_memory_profile(*args, **kwargs)


def start_server(*args: Any, **kwargs: Any) -> Any:
    """Starts the profiler server on port `port`."""
    return _ops.start_server(*args, **kwargs)


def start_trace(*args: Any, **kwargs: Any) -> Any:
    """Starts a profiler trace."""
    return _ops.start_trace(*args, **kwargs)


def stop_server(*args: Any, **kwargs: Any) -> Any:
    """Stops the running profiler server."""
    return _ops.stop_server(*args, **kwargs)


def stop_trace(*args: Any, **kwargs: Any) -> Any:
    """Stops the currently-running profiler trace."""
    return _ops.stop_trace(*args, **kwargs)


def trace(*args: Any, **kwargs: Any) -> Any:
    """Context manager to take a profiler trace."""
    return _ops.trace(*args, **kwargs)
