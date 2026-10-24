"""Print options configuration."""

from __future__ import annotations

import contextlib

from zero_jax._types import ArrayLikeSpec


def set_printoptions(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> None:
    """Set print options.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.
    """


@contextlib.contextmanager
def printoptions(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for setting print options.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    yield
