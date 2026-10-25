"""Frontend API routing for jax.scipy.stats.beta."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where
from zero_jax.scipy.special import betainc, betaln, xlog1py, xlogy


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        x = asarray(args[0], dtype=float32)
        a = asarray(args[1], dtype=float32)
        b = asarray(args[2], dtype=float32)
        loc = asarray(args[3] if len(args) > 3 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[4] if len(args) > 4 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        val = xlogy(a - 1.0, y) + xlog1py(b - 1.0, -y) - betaln(a, b) - log(scale)
        return where((y > 0.0) & (y < 1.0), val, float("-inf"))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        x = asarray(args[0], dtype=float32)
        a = asarray(args[1], dtype=float32)
        b = asarray(args[2], dtype=float32)
        loc = asarray(args[3] if len(args) > 3 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[4] if len(args) > 4 else kwargs.get("scale", 1.0), dtype=float32)
        y = (x - loc) / scale
        return betainc(a, b, y)
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta log cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return log(cdf(*args, **kwargs))
    return _ops.logcdf(*args, **kwargs)


def sf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta distribution survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return 1.0 - cdf(*args, **kwargs)
    return _ops.sf(*args, **kwargs)


def logsf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta distribution log survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``a``, ``b``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return log(sf(*args, **kwargs))
    return _ops.logsf(*args, **kwargs)
