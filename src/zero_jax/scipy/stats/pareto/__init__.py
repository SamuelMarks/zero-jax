"""Frontend API routing for jax.scipy.stats.pareto."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, where


def logpdf(x: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Pareto log probability distribution function.

    Args:
        x: Input quantiles.
        b: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    y = (x_arr - loc_arr) / scale_arr
    return where(y >= 1.0, log(b_arr) - (b_arr + 1.0) * log(y) - log(scale_arr), float("-inf"))


def pdf(x: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Pareto probability distribution function.

    Args:
        x: Input quantiles.
        b: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, b, loc, scale))


__all__ = [
    "logpdf",
    "pdf",
]
