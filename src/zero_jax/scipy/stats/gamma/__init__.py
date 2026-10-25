"""Frontend API routing for jax.scipy.stats.gamma."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where
from zero_jax.scipy.special import gammainc, gammaincc, gammaln, xlogy


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        a = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        val = xlogy(a - 1.0, y) - y - gammaln(a) - log(scale)
        return where(y >= 0.0, val, float("-inf"))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        a = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        return gammainc(a, y)
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma log cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return log(cdf(*args, **kwargs))
    return _ops.logcdf(*args, **kwargs)


def sf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        a = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        return gammaincc(a, y)
    return _ops.sf(*args, **kwargs)


def logsf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Gamma log survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return log(sf(*args, **kwargs))
    return _ops.logsf(*args, **kwargs)
