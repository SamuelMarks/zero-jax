"""Frontend API routing for jax.typing."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def ArrayLike(*args: Any, **kwargs: Any) -> Any:
    """Type annotation for JAX array-like objects."""
    return getattr(_ops, "ArrayLike")(*args, **kwargs)


def DTypeLike(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DTypeLike."""
    return getattr(_ops, "DTypeLike")(*args, **kwargs)
