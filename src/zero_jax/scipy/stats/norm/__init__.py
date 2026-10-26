"""Frontend API routing for jax.scipy.stats.norm."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
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


def logpdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal log probability distribution function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    z, s = _norm_z(x, loc, scale)
    return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - log(s)


def pdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal probability distribution function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, loc, scale))


def cdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal cumulative distribution function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    z, _ = _norm_z(x, loc, scale)
    return ndtr(z)


def logcdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal log cumulative distribution function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    z, _ = _norm_z(x, loc, scale)
    return log_ndtr(z)


def sf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal distribution survival function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    z, _ = _norm_z(x, loc, scale)
    return ndtr(-z)


def logsf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal distribution log survival function.

    Args:
        x: Input quantiles.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    z, _ = _norm_z(x, loc, scale)
    return log_ndtr(-z)


def ppf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal distribution percent point function.

    Args:
        q: Lower tail probability.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Quantile corresponding to ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return ndtri(q_arr) * scale_arr + loc_arr


def isf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Normal distribution inverse survival function.

    Args:
        q: Upper tail probability.
        loc: Distribution mean.
        scale: Distribution standard deviation.

    Returns:
        ArrayLikeSpec: Quantile corresponding to upper tail ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return -ndtri(q_arr) * scale_arr + loc_arr


__all__ = [
    "_norm_z",
    "cdf",
    "isf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "ppf",
    "sf",
]
