"""Frontend API routing for jax.scipy.stats.laplace."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import abs as _abs
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where


def logpdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Laplace log probability distribution function.

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
    return -_abs(x_arr - loc_arr) / scale_arr - math.log(2.0) - log(scale_arr)


def pdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Laplace probability distribution function.

    Args:
        x: Input quantiles.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, loc, scale))


def cdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Laplace cumulative distribution function.

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
    z = (x_arr - loc_arr) / scale_arr
    return where(z <= 0.0, 0.5 * exp(z), 1.0 - 0.5 * exp(-z))


__all__ = [
    "cdf",
    "logpdf",
    "pdf",
]
