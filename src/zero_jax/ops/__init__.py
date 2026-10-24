"""Frontend API routing for jax.ops."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def segment_max(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the maximum within segments of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.segment_max(*args, **kwargs)


def segment_min(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the minimum within segments of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.segment_min(*args, **kwargs)


def segment_prod(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the product within segments of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.segment_prod(*args, **kwargs)


def segment_sum(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the sum within segments of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.segment_sum(*args, **kwargs)
