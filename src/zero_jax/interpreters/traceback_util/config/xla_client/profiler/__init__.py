"""Module profiler."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue


class ProfileOptions:
    """Stub class for ProfileOptions."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ProfileOptions.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ProfilerServer:
    """Stub class for ProfilerServer."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ProfilerServer.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ProfilerSession:
    """Stub class for ProfilerSession."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ProfilerSession.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class TraceMe:
    """Stub class for TraceMe."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize TraceMe.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def aggregate_profiled_instructions(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for aggregate_profiled_instructions.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def get_fdo_profile(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for get_fdo_profile.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def get_profiled_instructions_proto(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for get_profiled_instructions_proto.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def register_plugin_profiler(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for register_plugin_profiler.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def start_server(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route start_server to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.start_server(*args, **kwargs)
