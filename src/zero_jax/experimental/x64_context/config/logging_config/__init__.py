"""Frontend API routing for jax.experimental.x64_context.config.logging_config."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def disable_all_debug_logging(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Disables all debug logging enabled via `enable_debug_logging`.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.disable_all_debug_logging(*args, **kwargs)


def enable_debug_logging(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Makes the specified logger log everything to stderr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_debug_logging(*args, **kwargs)


from . import logging, sys
