"""Frontend API routing for jax.scipy.stats.t."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, log1p
from zero_jax.scipy.special import gammaln


def logpdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Student's T log probability distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    return gammaln((df_arr + 1.0) * 0.5) - gammaln(df_arr * 0.5) - 0.5 * log(df_arr * math.pi) - log(scale_arr) - 0.5 * (df_arr + 1.0) * log1p((z * z) / df_arr)


def pdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Student's T probability distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, df, loc, scale))


__all__ = [
    "logpdf",
    "pdf",
]
