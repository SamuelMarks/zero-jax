"""Frontend API routing for jax.ops."""

from typing import Any


def segment_max(*args: Any, **kwargs: Any) -> Any:
    """Computes the maximum within segments of an array."""
    raise NotImplementedError("segment_max not yet implemented in zero-jax")


def segment_min(*args: Any, **kwargs: Any) -> Any:
    """Computes the minimum within segments of an array."""
    raise NotImplementedError("segment_min not yet implemented in zero-jax")


def segment_prod(*args: Any, **kwargs: Any) -> Any:
    """Computes the product within segments of an array."""
    raise NotImplementedError("segment_prod not yet implemented in zero-jax")


def segment_sum(*args: Any, **kwargs: Any) -> Any:
    """Computes the sum within segments of an array."""
    raise NotImplementedError("segment_sum not yet implemented in zero-jax")
