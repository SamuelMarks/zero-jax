"""Frontend API routing for jax.scipy.stats.logistic."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log
from zero_jax.scipy.special import expit, logit


def cdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic cumulative distribution function.

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
    return expit((x_arr - loc_arr) / scale_arr)


def logpdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic log probability distribution function.

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
    return -z - 2.0 * log(1.0 + exp(-z)) - log(scale_arr)


def pdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic probability distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, loc, scale))


def sf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic distribution survival function.

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
    return expit(-(x_arr - loc_arr) / scale_arr)


def ppf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic distribution percent point function.

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
    return loc_arr + scale_arr * logit(q_arr)


def isf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Logistic distribution inverse survival function.

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
    return loc_arr - scale_arr * logit(q_arr)


__all__ = [
    "cdf",
    "isf",
    "logpdf",
    "pdf",
    "ppf",
    "sf",
]
