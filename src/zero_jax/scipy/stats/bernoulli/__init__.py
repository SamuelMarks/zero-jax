"""Frontend API routing for jax.scipy.stats.bernoulli."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, floor, ones_like, where, zeros_like
from zero_jax.scipy.special import xlog1py, xlogy


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli log probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        k = asarray(args[0], dtype=float32) - asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        p = asarray(args[1], dtype=float32)
        val = xlogy(k, p) + xlog1py(1.0 - k, -p)
        return where((k == 0.0) | (k == 1.0), val, float("-inf"))
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``k``, ``p``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        k = floor(asarray(args[0], dtype=float32))
        p = asarray(args[1], dtype=float32)
        return where(k < 0.0, zeros_like(k), where(k >= 1.0, ones_like(k), 1.0 - p))
    return _ops.cdf(*args, **kwargs)


def ppf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli percent point function.

    Args:
        *args: Input parameter ``args`` (``q``, ``p``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        q = asarray(args[0], dtype=float32)
        p = asarray(args[1], dtype=float32)
        return where((q < 0.0) | (q > 1.0), float("nan"), where(q <= 1.0 - p, zeros_like(q), ones_like(q)))
    return _ops.ppf(*args, **kwargs)
