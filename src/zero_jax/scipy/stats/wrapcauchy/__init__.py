"""Frontend API routing for jax.scipy.stats.wrapcauchy."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, cos, exp, float32, log


def logpdf(x: ArrayLikeSpec, c: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wrapped Cauchy log probability distribution function.

    Args:
        x: Input angle quantiles in radians.
        c: Shape concentration parameter in ``(0, 1)``.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    c_arr = asarray(c, dtype=float32)
    return log(1.0 - c_arr * c_arr) - math.log(2.0 * math.pi) - log(1.0 + c_arr * c_arr - 2.0 * c_arr * cos(x_arr))


def pdf(x: ArrayLikeSpec, c: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wrapped Cauchy probability distribution function.

    Args:
        x: Input angle quantiles in radians.
        c: Shape concentration parameter in ``(0, 1)``.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, c))


__all__ = [
    "logpdf",
    "pdf",
]
