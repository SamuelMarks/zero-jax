"""Module transfer_guard_lib."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue


class TransferGuardLevel:
    """Stub class for TransferGuardLevel."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize TransferGuardLevel.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class TransferGuardState:
    """Stub class for TransferGuardState."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize TransferGuardState.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def global_state(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route global_state to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.global_state(*args, **kwargs)


def thread_local_state(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route thread_local_state to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.thread_local_state(*args, **kwargs)
