"""Frontend API routing for jax.distributed."""

from typing import Any


def initialize(*args: Any, **kwargs: Any) -> Any:
    """Initializes the JAX distributed system."""
    raise NotImplementedError("initialize not yet implemented in zero-jax")


def shutdown(*args: Any, **kwargs: Any) -> Any:
    """Shuts down the distributed system."""
    raise NotImplementedError("shutdown not yet implemented in zero-jax")
