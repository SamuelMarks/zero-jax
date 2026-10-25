"""Frontend API routing for jax.scipy.stats.cauchy."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import arctan, asarray, exp, float32, log, log1p, tan


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy log probability distribution function.

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
        return -math.log(math.pi) - log(scale) - log1p(z * z)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy probability distribution function.

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
    """Cauchy cumulative distribution function.

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
        return 0.5 + arctan((x - loc) / scale) / math.pi
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy log cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return log(cdf(*args, **kwargs))
    return _ops.logcdf(*args, **kwargs)


def sf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy distribution survival function.

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
        return 0.5 - arctan((x - loc) / scale) / math.pi
    return _ops.sf(*args, **kwargs)


def logsf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy distribution log survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return log(sf(*args, **kwargs))
    return _ops.logsf(*args, **kwargs)


def ppf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy distribution percent point function.

    Args:
        *args: Input parameter ``args`` (``q``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        q = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        return loc + scale * tan(math.pi * (q - 0.5))
    return _ops.ppf(*args, **kwargs)


def isf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cauchy distribution inverse survival function.

    Args:
        *args: Input parameter ``args`` (``q``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        q = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        return loc + scale * tan(math.pi * (0.5 - q))
    return _ops.isf(*args, **kwargs)
