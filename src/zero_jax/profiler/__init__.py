"""Frontend API routing for jax.profiler."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class StepTraceAnnotation:
    """Context manager that generates a step trace event in the profiler."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name if name is not None else (args[0] if args else "step")
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs

    def __enter__(self) -> StepTraceAnnotation:
        """Enter step trace context.

        Returns:
            StepTraceAnnotation: ``self``.
        """
        return self

    def __exit__(self, *exc_info: ConfigValue) -> None:
        """Exit step trace context.

        Args:
            *exc_info: Exception info tuple.
        """


class TraceAnnotation:
    """Context manager that generates a trace event in the profiler."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name if name is not None else (args[0] if args else "trace")
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs

    def __enter__(self) -> TraceAnnotation:
        """Enter trace context.

        Returns:
            TraceAnnotation: ``self``.
        """
        return self

    def __exit__(self, *exc_info: ConfigValue) -> None:
        """Exit trace context.

        Args:
            *exc_info: Exception info tuple.
        """


def annotate_function(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Decorator that generates a trace event for the execution of a function.

    Args:
        *args: Input parameter ``args`` (``func``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and callable(args[0]):
        return args[0]
    return _ops.annotate_function(*args, **kwargs)


def device_memory_profile(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Captures a JAX device memory profile as ``pprof``-format protocol buffer.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args or kwargs:
        return b""
    return _ops.device_memory_profile(*args, **kwargs)


def save_device_memory_profile(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Collects a device memory profile and writes it to a file.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return None
    return _ops.save_device_memory_profile(*args, **kwargs)


def start_server(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Starts the profiler server on port `port`.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return None
    return _ops.start_server(*args, **kwargs)


def start_trace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Starts a profiler trace.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return None
    return _ops.start_trace(*args, **kwargs)


def stop_server(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stops the running profiler server.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return None
    return _ops.stop_server(*args, **kwargs)


def stop_trace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stops the currently-running profiler trace.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return None
    return _ops.stop_trace(*args, **kwargs)


@contextlib.contextmanager
def _trace_ctx() -> Iterator[None]:
    """No-op profiler trace context manager.

    Yields:
        None: Context yield.
    """
    yield


def trace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager to take a profiler trace.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return _trace_ctx()
    return _ops.trace(*args, **kwargs)
