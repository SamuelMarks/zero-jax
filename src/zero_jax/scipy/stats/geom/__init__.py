"""Frontend API routing for jax.scipy.stats.geom."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, log1p, where


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Geometric log probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        k = asarray(args[0], dtype=float32) - asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        p = asarray(args[1], dtype=float32)
        val = (k - 1.0) * log1p(-p) + log(p)
        return where(k >= 1.0, val, float("-inf"))
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Geometric probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)
