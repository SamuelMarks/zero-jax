"""Frontend API routing for jax.scipy.stats.chi2."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, float32
from zero_jax.scipy.stats import gamma as _gamma


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.logpdf(x, df * 0.5, loc, scale * 2.0)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.pdf(x, df * 0.5, loc, scale * 2.0)
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.cdf(x, df * 0.5, loc, scale * 2.0)
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square log cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.logcdf(x, df * 0.5, loc, scale * 2.0)
    return _ops.logcdf(*args, **kwargs)


def sf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.sf(x, df * 0.5, loc, scale * 2.0)
    return _ops.sf(*args, **kwargs)


def logsf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Chi-square log survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        return _gamma.logsf(x, df * 0.5, loc, scale * 2.0)
    return _ops.logsf(*args, **kwargs)
