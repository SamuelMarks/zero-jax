"""Frontend API routing for jax.distributed."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue

_INITIALIZED: bool = False


def initialize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Initializes the JAX distributed system.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    global _INITIALIZED
    _INITIALIZED = True
    if not args and not kwargs:
        try:
            return _ops.initialize()
        except AttributeError:
            return None
    return None


def is_initialized() -> bool:
    """Check whether the JAX distributed system is initialized.

    Returns:
        bool: ``True`` if initialized.
    """
    return _INITIALIZED


def shutdown(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Shuts down the distributed system.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    global _INITIALIZED
    _INITIALIZED = False
    if not args and not kwargs:
        try:
            return _ops.shutdown()
        except AttributeError:
            return None
    return None
