"""Frontend API routing for jax.scipy.stats.expon."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponential log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        z = (x - loc) / scale
        return where(x < loc, float("-inf"), -z - log(scale))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponential probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
