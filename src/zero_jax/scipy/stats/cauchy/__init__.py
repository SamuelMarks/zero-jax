"""Frontend API routing for jax.scipy.stats.cauchy."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import arctan, asarray, exp, float32, log, log1p, tan


def logpdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy log probability distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    return -math.log(math.pi) - log(scale_arr) - log1p(z * z)


def pdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy probability distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, loc, scale))


def cdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy cumulative distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return 0.5 + arctan((x_arr - loc_arr) / scale_arr) / math.pi


def logcdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy log cumulative distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    return log(cdf(x, loc, scale))


def sf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy distribution survival function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return 0.5 - arctan((x_arr - loc_arr) / scale_arr) / math.pi


def logsf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy distribution log survival function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    return log(sf(x, loc, scale))


def ppf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy distribution percent point function.

    Args:
        q: Lower tail probability.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Quantile corresponding to ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return loc_arr + scale_arr * tan(math.pi * (q_arr - 0.5))


def isf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Cauchy distribution inverse survival function.

    Args:
        q: Upper tail probability.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Quantile corresponding to upper tail ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return loc_arr + scale_arr * tan(math.pi * (0.5 - q_arr))


__all__ = [
    "cdf",
    "isf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "ppf",
    "sf",
]
