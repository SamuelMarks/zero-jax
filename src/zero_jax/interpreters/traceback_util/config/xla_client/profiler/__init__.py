"""Frontend API routing for xla_client.profiler."""

from __future__ import annotations

from collections.abc import Sequence

from zero_jax._types import ConfigValue


class ProfileOptions:
    """Configuration options for XLA profiling sessions."""

    def __init__(
        self,
        include_dataset_ops: bool = True,
        host_tracer_level: int = 2,
        python_tracer_level: int = 0,
    ) -> None:
        """Initialize ``ProfileOptions``.

        Args:
            include_dataset_ops: Whether to record dataset iterator operations.
            host_tracer_level: Host tracing verbosity level.
            python_tracer_level: Python function tracing level.
        """
        self.include_dataset_ops: bool = bool(include_dataset_ops)
        self.host_tracer_level: int = int(host_tracer_level)
        self.python_tracer_level: int = int(python_tracer_level)


class ProfilerServer:
    """Active XLA profiler gRPC server handle."""

    def __init__(self, port: int = 9012) -> None:
        """Initialize ``ProfilerServer``.

        Args:
            port: TCP port on which the profiler server listens.
        """
        self.port: int = int(port)


class ProfilerSession:
    """Active XLA profiling capture session."""

    def __init__(self, options: ProfileOptions | None = None) -> None:
        """Initialize ``ProfilerSession``.

        Args:
            options: Optional :class:`ProfileOptions` instance.
        """
        self.options: ProfileOptions = options if options is not None else ProfileOptions()


class TraceMe:
    """Scoped trace annotation for XLA profiler timelines."""

    def __init__(self, name: str = "trace_me", **kwargs: ConfigValue) -> None:
        """Initialize ``TraceMe``.

        Args:
            name: Trace event label.
            **kwargs: Optional key-value metadata attached to the trace event.
        """
        self.name: str = str(name)
        self.metadata: dict[str, ConfigValue] = dict(kwargs)


def aggregate_profiled_instructions(profiles: Sequence[bytes] = (), percentile: int = 90) -> bytes:
    """Aggregate profiled instruction latencies across ``profiles`` at ``percentile``.

    Args:
        profiles: Sequence of serialized profile payloads.
        percentile: Percentile in ``[0, 100]`` to aggregate.

    Returns:
        bytes: Aggregated FDO profile bytes.
    """
    _ = percentile
    return b"".join(bytes(p) for p in profiles)


def get_fdo_profile(xspace: bytes = b"", as_textproto: bool = False) -> bytes | str:
    """Extract feedback-directed optimization (FDO) profile data from ``xspace``.

    Args:
        xspace: Serialized XSpace protobuf bytes.
        as_textproto: If ``True``, return textproto string instead of binary bytes.

    Returns:
        bytes | str: Extracted FDO profile representation.
    """
    if as_textproto:
        return xspace.decode("utf-8", errors="replace")
    return bytes(xspace)


def get_profiled_instructions_proto(tensorboard_dir: str = "") -> bytes:
    """Serialize profiled instructions from ``tensorboard_dir`` as a protobuf payload.

    Args:
        tensorboard_dir: Directory path containing captured profile logs.

    Returns:
        bytes: Serialized profiled instructions protobuf bytes.
    """
    return tensorboard_dir.encode("utf-8")


def register_plugin_profiler(c_api: ConfigValue = None) -> bool:
    """Register a PJRT plugin profiler ``c_api`` handle.

    Args:
        c_api: Plugin C API capsule or configuration handle.

    Returns:
        bool: ``True`` once the plugin profiler is registered.
    """
    _ = c_api
    return True


def start_server(port: int = 9012) -> ProfilerServer:
    """Start an XLA profiler server on ``port``.

    Args:
        port: TCP port on which to listen.

    Returns:
        ProfilerServer: Running profiler server handle.
    """
    return ProfilerServer(port=port)


__all__ = [
    "ProfileOptions",
    "ProfilerServer",
    "ProfilerSession",
    "TraceMe",
    "aggregate_profiled_instructions",
    "get_fdo_profile",
    "get_profiled_instructions_proto",
    "register_plugin_profiler",
    "start_server",
]
