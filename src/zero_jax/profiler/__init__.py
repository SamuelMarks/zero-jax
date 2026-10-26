"""Frontend API routing for jax.profiler."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable


class StepTraceAnnotation:
    """Context manager that generates a step trace event in the profiler."""

    def __init__(
        self,
        name: str = "step",
        *,
        step_num: int | None = None,
        id: ConfigValue = None,
        value: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``StepTraceAnnotation``.

        Args:
            name: Name of the step trace event.
            step_num: Optional step number.
            id: Optional identifier metadata.
            value: Optional value metadata.
            **kwargs: Additional keyword metadata.
        """
        self.name = name
        self.step_num = step_num
        self.id = id
        self.value = value
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

    def __init__(
        self,
        name: str = "trace",
        *,
        id: ConfigValue = None,
        value: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``TraceAnnotation``.

        Args:
            name: Name of the trace event.
            id: Optional identifier metadata.
            value: Optional value metadata.
            **kwargs: Additional keyword metadata.
        """
        self.name = name
        self.id = id
        self.value = value
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


def annotate_function(func: VariadicCallable[ArrayLikeSpec], name: str | None = None, **decorator_kwargs: ConfigValue) -> VariadicCallable[ArrayLikeSpec]:
    """Decorator that generates a trace event for the execution of a function.

    Args:
        func: Function to annotate.
        name: Optional event name override.
        **decorator_kwargs: Additional trace metadata.

    Returns:
        VariadicCallable[ArrayLikeSpec]: The annotated function.
    """
    _ = (name, decorator_kwargs)
    return func


def device_memory_profile(backend: str | None = None) -> bytes:
    """Capture a JAX device memory profile as ``pprof``-format protocol buffer bytes.

    Args:
        backend: Optional backend name.

    Returns:
        bytes: Encoded memory profile bytes.
    """
    _ = backend
    return b""


def save_device_memory_profile(filename: str, backend: str | None = None) -> None:
    """Collect a device memory profile and write it to ``filename``.

    Args:
        filename: Destination file path.
        backend: Optional backend name.
    """
    _ = (filename, backend)


def start_server(port: int) -> None:
    """Start the profiler server on ``port``.

    Args:
        port: Port number to bind.
    """
    _ = port


def start_trace(log_dir: str, create_perfetto_link: bool = False, create_perfetto_trace: bool = False) -> None:
    """Start a profiler trace writing to ``log_dir``.

    Args:
        log_dir: Directory where trace files are written.
        create_perfetto_link: Whether to generate a Perfetto UI link.
        create_perfetto_trace: Whether to generate a Perfetto trace file.
    """
    _ = (log_dir, create_perfetto_link, create_perfetto_trace)


def stop_server() -> None:
    """Stop the running profiler server."""


def stop_trace() -> None:
    """Stop the currently-running profiler trace."""


@contextlib.contextmanager
def trace(log_dir: str, create_perfetto_link: bool = False, create_perfetto_trace: bool = False) -> Iterator[None]:
    """Context manager to take a profiler trace.

    Args:
        log_dir: Directory where trace files are written.
        create_perfetto_link: Whether to generate a Perfetto UI link.
        create_perfetto_trace: Whether to generate a Perfetto trace file.

    Yields:
        None: Profiler trace context.
    """
    start_trace(log_dir, create_perfetto_link=create_perfetto_link, create_perfetto_trace=create_perfetto_trace)
    try:
        yield
    finally:
        stop_trace()


__all__ = [
    "StepTraceAnnotation",
    "TraceAnnotation",
    "annotate_function",
    "device_memory_profile",
    "save_device_memory_profile",
    "start_server",
    "start_trace",
    "stop_server",
    "stop_trace",
    "trace",
]
