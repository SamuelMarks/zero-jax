"""Frontend API routing for jax.scipy.stats.laplace."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import abs as _abs
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Laplace log probability distribution function.

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
        return -_abs(x - loc) / scale - math.log(2.0) - log(scale)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Laplace probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Laplace cumulative distribution function.

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
        return where(z <= 0.0, 0.5 * exp(z), 1.0 - 0.5 * exp(-z))
    return _ops.cdf(*args, **kwargs)
