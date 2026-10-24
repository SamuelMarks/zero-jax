"""Module jax_jit."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class JitState:
    """Frontend class for ``JitState``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class PyArgSignature:
    """Stub class for PyArgSignature."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PyArgSignature.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def get_enable_x64(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route get_enable_x64 to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.get_enable_x64(*args, **kwargs)


def global_state(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route global_state to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.global_state(*args, **kwargs)


def set_thread_local_state_initialization_callback(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route set_thread_local_state_initialization_callback to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.set_thread_local_state_initialization_callback(*args, **kwargs)


def swap_thread_local_state_disable_jit(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route swap_thread_local_state_disable_jit to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.swap_thread_local_state_disable_jit(*args, **kwargs)


def thread_local_state(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route thread_local_state to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.thread_local_state(*args, **kwargs)
