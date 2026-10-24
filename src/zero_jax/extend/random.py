"""Random submodule for zero_jax.extend."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec


def stub(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


__all__ = ["stub"]
