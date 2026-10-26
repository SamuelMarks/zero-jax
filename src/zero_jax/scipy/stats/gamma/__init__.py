"""Frontend API routing for jax.scipy.stats.gamma."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where
from zero_jax.scipy.special import gammainc, gammaincc, gammaln, xlogy


def logpdf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma log probability distribution function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    val = xlogy(a_arr - 1.0, y) - y - gammaln(a_arr) - log(scale_arr)
    return where(y >= 0.0, val, float("-inf"))


def pdf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma probability distribution function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, a, loc, scale))


def cdf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma cumulative distribution function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    return gammainc(a_arr, y)


def logcdf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma log cumulative distribution function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    return log(cdf(x, a, loc, scale))


def sf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma survival function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    return gammaincc(a_arr, y)


def logsf(x: ArrayLikeSpec, a: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Gamma log survival function.

    Args:
        x: Input quantiles.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    return log(sf(x, a, loc, scale))


__all__ = [
    "cdf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "sf",
]
