"""Module ifrt_programs."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue


class CompileOptions:
    """Stub class for CompileOptions."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize CompileOptions.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Program:
    """Stub class for Program."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Program.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def make_hlo_program(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for make_hlo_program.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def make_plugin_compile_options(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for make_plugin_compile_options.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def make_plugin_program(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for make_plugin_program.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def make_xla_compile_options(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for make_xla_compile_options.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None
