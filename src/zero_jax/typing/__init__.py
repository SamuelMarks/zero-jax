"""Frontend API routing for jax.typing."""

from typing import Any


def ArrayLike(*args: Any, **kwargs: Any) -> Any:
    """Type annotation for JAX array-like objects."""
    raise NotImplementedError("ArrayLike not yet implemented in zero-jax")


def DTypeLike(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DTypeLike."""
    raise NotImplementedError("DTypeLike not yet implemented in zero-jax")
