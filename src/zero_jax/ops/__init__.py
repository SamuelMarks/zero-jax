"""Frontend API routing for jax.ops."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def segment_max(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the maximum within segments of an array.

    Args:
        *args: Input parameter ``args`` (``data``, ``segment_ids``, ``num_segments=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return _wrap(_ops.segment_max(_to_tensor(args[0]), _to_tensor(args[1]), *args[2:], **kwargs))
    return _ops.segment_max(*args, **kwargs)


def segment_min(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the minimum within segments of an array.

    Args:
        *args: Input parameter ``args`` (``data``, ``segment_ids``, ``num_segments=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return _wrap(_ops.segment_min(_to_tensor(args[0]), _to_tensor(args[1]), *args[2:], **kwargs))
    return _ops.segment_min(*args, **kwargs)


def segment_prod(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the product within segments of an array.

    Args:
        *args: Input parameter ``args`` (``data``, ``segment_ids``, ``num_segments=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return _wrap(_ops.segment_prod(_to_tensor(args[0]), _to_tensor(args[1]), *args[2:], **kwargs))
    return _ops.segment_prod(*args, **kwargs)


def segment_sum(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the sum within segments of an array.

    Args:
        *args: Input parameter ``args`` (``data``, ``segment_ids``, ``num_segments=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return _wrap(_ops.segment_sum(_to_tensor(args[0]), _to_tensor(args[1]), *args[2:], **kwargs))
    return _ops.segment_sum(*args, **kwargs)
