"""Frontend API routing for jax.ops."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def segment_max(*args: Any, **kwargs: Any) -> Any:
    """Computes the maximum within segments of an array."""
    return _ops.segment_max(*args, **kwargs)


def segment_min(*args: Any, **kwargs: Any) -> Any:
    """Computes the minimum within segments of an array."""
    return _ops.segment_min(*args, **kwargs)


def segment_prod(*args: Any, **kwargs: Any) -> Any:
    """Computes the product within segments of an array."""
    return _ops.segment_prod(*args, **kwargs)


def segment_sum(*args: Any, **kwargs: Any) -> Any:
    """Computes the sum within segments of an array."""
    return _ops.segment_sum(*args, **kwargs)
