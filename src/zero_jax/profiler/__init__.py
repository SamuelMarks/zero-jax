"""Frontend API routing for jax.profiler."""

from typing import Any


class StepTraceAnnotation:
    """Context manager that generates a step trace event in the profiler."""

    pass


def annotate_function(*args: Any, **kwargs: Any) -> Any:
    """Decorator that generates a trace event for the execution of a function."""
    raise NotImplementedError("annotate_function not yet implemented in zero-jax")


def device_memory_profile(*args: Any, **kwargs: Any) -> Any:
    """Captures a JAX device memory profile as ``pprof``-format protocol buffer."""
    raise NotImplementedError("device_memory_profile not yet implemented in zero-jax")


def save_device_memory_profile(*args: Any, **kwargs: Any) -> Any:
    """Collects a device memory profile and writes it to a file."""
    raise NotImplementedError(
        "save_device_memory_profile not yet implemented in zero-jax"
    )


def start_server(*args: Any, **kwargs: Any) -> Any:
    """Starts the profiler server on port `port`."""
    raise NotImplementedError("start_server not yet implemented in zero-jax")


def start_trace(*args: Any, **kwargs: Any) -> Any:
    """Starts a profiler trace."""
    raise NotImplementedError("start_trace not yet implemented in zero-jax")


def stop_server(*args: Any, **kwargs: Any) -> Any:
    """Stops the running profiler server."""
    raise NotImplementedError("stop_server not yet implemented in zero-jax")


def stop_trace(*args: Any, **kwargs: Any) -> Any:
    """Stops the currently-running profiler trace."""
    raise NotImplementedError("stop_trace not yet implemented in zero-jax")


def trace(*args: Any, **kwargs: Any) -> Any:
    """Context manager to take a profiler trace."""
    raise NotImplementedError("trace not yet implemented in zero-jax")
