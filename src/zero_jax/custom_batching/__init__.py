"""Frontend API routing for jax.custom_batching."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class custom_vmap:
    """Mock implementation for custom_vmap."""

    pass


def sequential_vmap(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for sequential_vmap."""
    return getattr(_ops, "sequential_vmap")(*args, **kwargs)
