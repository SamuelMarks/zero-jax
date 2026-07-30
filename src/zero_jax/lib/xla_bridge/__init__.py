"""Frontend API routing for jax.lib.xla_bridge."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops

from . import xla_client


def default_backend(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for default_backend.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.default_backend(*args, **kwargs)


def get_backend(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for get_backend.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.get_backend(*args, **kwargs)


def get_compile_options(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for get_compile_options.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.get_compile_options(*args, **kwargs)


__all__ = ["default_backend", "get_backend", "get_compile_options", "xla_client"]
