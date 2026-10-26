"""Frontend API routing for jax.scipy.stats.uniform."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, ones_like, where, zeros_like


def logpdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Uniform log probability distribution function.

    Args:
        x: Input quantiles.
        loc: Lower endpoint of the uniform interval.
        scale: Width of the uniform interval.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return where((x_arr >= loc_arr) & (x_arr <= loc_arr + scale_arr), -log(scale_arr) * ones_like(x_arr), float("-inf"))


def pdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Uniform probability distribution function.

    Args:
        x: Input quantiles.
        loc: Lower endpoint of the uniform interval.
        scale: Width of the uniform interval.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, loc, scale))


def cdf(x: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Uniform cumulative distribution function.

    Args:
        x: Input quantiles.
        loc: Lower endpoint of the uniform interval.
        scale: Width of the uniform interval.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    return where(z < 0.0, zeros_like(z), where(z > 1.0, ones_like(z), z))


def ppf(q: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Uniform distribution percent point function.

    Args:
        q: Lower tail probability.
        loc: Lower endpoint of the uniform interval.
        scale: Width of the uniform interval.

    Returns:
        ArrayLikeSpec: Quantile corresponding to ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return where((q_arr >= 0.0) & (q_arr <= 1.0), loc_arr + q_arr * scale_arr, float("nan"))


__all__ = [
    "cdf",
    "logpdf",
    "pdf",
    "ppf",
]
