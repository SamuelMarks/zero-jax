"""Frontend API routing for jax.scipy.stats.norm."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log
from zero_jax.scipy.special import log_ndtr, ndtr, ndtri


def _norm_z(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Standardize ``x`` by ``loc`` and ``scale``.

    Args:
        x: Input quantiles.
        loc: Mean parameter.
        scale: Standard deviation parameter.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple of ``(z, scale_arr)``.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return (x_arr - loc_arr) / scale_arr, scale_arr


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = args[0]
        loc = args[1] if len(args) > 1 else kwargs.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1.0)
        z, s = _norm_z(x, loc, scale)
        return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - log(s)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal probability distribution function.

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
    """Normal cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = args[0]
        loc = args[1] if len(args) > 1 else kwargs.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1.0)
        z, _ = _norm_z(x, loc, scale)
        return ndtr(z)
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal log cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = args[0]
        loc = args[1] if len(args) > 1 else kwargs.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1.0)
        z, _ = _norm_z(x, loc, scale)
        return log_ndtr(z)
    return _ops.logcdf(*args, **kwargs)


def sf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal distribution survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = args[0]
        loc = args[1] if len(args) > 1 else kwargs.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1.0)
        z, _ = _norm_z(x, loc, scale)
        return ndtr(-z)
    return _ops.sf(*args, **kwargs)


def logsf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal distribution log survival function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = args[0]
        loc = args[1] if len(args) > 1 else kwargs.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1.0)
        z, _ = _norm_z(x, loc, scale)
        return log_ndtr(-z)
    return _ops.logsf(*args, **kwargs)


def ppf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal distribution percent point function.

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
        return ndtri(q) * scale + loc
    return _ops.ppf(*args, **kwargs)


def isf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal distribution inverse survival function.

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
        return -ndtri(q) * scale + loc
    return _ops.isf(*args, **kwargs)
