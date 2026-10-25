"""Frontend API routing for jax.scipy.stats.pareto."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Pareto log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        b = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        return where(y >= 1.0, log(b) - (b + 1.0) * log(y) - log(scale), float("-inf"))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Pareto probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
