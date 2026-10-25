"""Frontend API routing for jax.scipy.stats.poisson."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, floor, where, zeros_like
from zero_jax.scipy.special import gammaincc, gammaln, xlogy


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Poisson log probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``mu``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        k = asarray(args[0], dtype=float32) - asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        mu = asarray(args[1], dtype=float32)
        val = xlogy(k, mu) - gammaln(k + 1.0) - mu
        return where(k >= 0.0, val, float("-inf"))
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Poisson probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``mu``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Poisson cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``k``, ``mu``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        k = floor(asarray(args[0], dtype=float32) - asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32))
        mu = asarray(args[1], dtype=float32)
        return where(k < 0.0, zeros_like(k), gammaincc(k + 1.0, mu))
    return _ops.cdf(*args, **kwargs)
