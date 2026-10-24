"""Module logging_config."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue


def disable_all_debug_logging(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route disable_all_debug_logging to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.disable_all_debug_logging(*args, **kwargs)


def enable_debug_logging(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route enable_debug_logging to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.enable_debug_logging(*args, **kwargs)
