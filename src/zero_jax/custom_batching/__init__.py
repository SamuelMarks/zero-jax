"""Frontend API routing for jax.custom_batching."""

from typing import Any


class custom_vmap:
    """Mock implementation for custom_vmap."""

    pass


def sequential_vmap(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for sequential_vmap."""
    raise NotImplementedError("sequential_vmap not yet implemented in zero-jax")
