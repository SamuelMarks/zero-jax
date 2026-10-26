"""Frontend API routing for jax.typing."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, DTypeSpec
from zero_jax.numpy.lax_numpy import asarray, dtype


def ArrayLike(x: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Convert or validate an array-like object.

    Args:
        x: Input array-like value.

    Returns:
        ArrayLikeSpec: Converted array-like object.
    """
    return asarray(x)


def DTypeLike(x: DTypeSpec = "float32") -> DTypeSpec:
    """Convert or validate a dtype-like object.

    Args:
        x: Input dtype specification.

    Returns:
        DTypeSpec: Canonical dtype object.
    """
    return dtype(x)


__all__ = [
    "ArrayLike",
    "DTypeLike",
]
