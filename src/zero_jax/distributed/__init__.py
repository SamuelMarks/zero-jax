"""Frontend API routing for jax.distributed."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def initialize(*args: Any, **kwargs: Any) -> Any:
    """Initializes the JAX distributed system."""
    return getattr(_ops, "initialize")(*args, **kwargs)


def shutdown(*args: Any, **kwargs: Any) -> Any:
    """Shuts down the distributed system."""
    return getattr(_ops, "shutdown")(*args, **kwargs)
