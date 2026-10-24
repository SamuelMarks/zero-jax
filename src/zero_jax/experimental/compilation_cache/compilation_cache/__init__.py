"""Frontend API routing for jax.experimental.compilation_cache.compilation_cache."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def initialize_cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """This API is deprecated; use set_cache_dir instead.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.initialize_cache(*args, **kwargs)


def is_initialized(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Deprecated.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.is_initialized(*args, **kwargs)


def reset_cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get back to pristine, uninitialized state.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.reset_cache(*args, **kwargs)


def set_cache_dir(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Sets the persistent compilation cache directory.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.set_cache_dir(*args, **kwargs)
