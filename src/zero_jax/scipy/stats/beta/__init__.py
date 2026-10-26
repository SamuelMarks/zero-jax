"""Frontend API routing for jax.scipy.stats.beta."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where
from zero_jax.scipy.special import betainc, betaln, xlog1py, xlogy


def logpdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta log probability distribution function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    val = xlogy(a_arr - 1.0, y) + xlog1py(b_arr - 1.0, -y) - betaln(a_arr, b_arr) - log(scale_arr)
    return where((y > 0.0) & (y < 1.0), val, float("-inf"))


def pdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta probability distribution function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, a, b, loc, scale))


def cdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta cumulative distribution function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    return betainc(a_arr, b_arr, y)


def logcdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta log cumulative distribution function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    return log(cdf(x, a, b, loc, scale))


def sf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta distribution survival function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    return 1.0 - cdf(x, a, b, loc, scale)


def logsf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Beta distribution log survival function.

    Args:
        x: Input quantiles.
        a: First shape parameter.
        b: Second shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    return log(sf(x, a, b, loc, scale))


__all__ = [
    "cdf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "sf",
]
