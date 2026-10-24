"""Frontend API routing for jax.version."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


def annotations(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for annotations.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


from . import datetime, os, pathlib, subprocess
