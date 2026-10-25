"""Frontend API routing for jax.lib.xla_bridge."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue

from . import xla_client


def default_backend(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for default_backend.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if not args and not kwargs:
        try:
            return _ops.default_backend()
        except AttributeError:
            from zero_jax.api.core import default_backend as _core_default_backend

            return _core_default_backend()
    return _ops.default_backend(*args, **kwargs)


def get_backend(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for get_backend.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if not args and not kwargs:
        try:
            return _ops.get_backend()
        except AttributeError:
            from zero_jax.lax import Device

            return Device()
    return _ops.get_backend(*args, **kwargs)


def get_compile_options(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for get_compile_options.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if not args and not kwargs:
        try:
            return _ops.get_compile_options()
        except AttributeError:
            return {}
    return _ops.get_compile_options(*args, **kwargs)


__all__ = ["default_backend", "get_backend", "get_compile_options", "xla_client"]
