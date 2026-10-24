"""Frontend API routing for jax.typing."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, DTypeSpec


def ArrayLike(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Type annotation and wrapper for JAX array-like objects.

    Args:
        *args: Positional arguments forwarded to the compiler proxy op.
        **kwargs: Keyword arguments forwarded to the compiler proxy op.

    Returns:
        ArrayLikeSpec: The result from the compiler proxy op.
    """
    return _ops.ArrayLike(*args, **kwargs)


def DTypeLike(*args: DTypeSpec, **kwargs: DTypeSpec) -> DTypeSpec:
    """Type annotation and wrapper for JAX dtype-like objects.

    Args:
        *args: Positional arguments forwarded to the compiler proxy op.
        **kwargs: Keyword arguments forwarded to the compiler proxy op.

    Returns:
        DTypeSpec: The result from the compiler proxy op.
    """
    return _ops.DTypeLike(*args, **kwargs)


__all__ = [
    "ArrayLike",
    "DTypeLike",
]
