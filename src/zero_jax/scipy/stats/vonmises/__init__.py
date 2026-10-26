"""Frontend API routing for jax.scipy.stats.vonmises."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, cos, exp, float32, log
from zero_jax.scipy.special import i0e


def logpdf(x: ArrayLikeSpec, kappa: ArrayLikeSpec) -> ArrayLikeSpec:
    """Von Mises log probability distribution function.

    Args:
        x: Input angle quantiles in radians.
        kappa: Concentration parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    kappa_arr = asarray(kappa, dtype=float32)
    return kappa_arr * (cos(x_arr) - 1.0) - math.log(2.0 * math.pi) - log(i0e(kappa_arr))


def pdf(x: ArrayLikeSpec, kappa: ArrayLikeSpec) -> ArrayLikeSpec:
    """Von Mises probability distribution function.

    Args:
        x: Input angle quantiles in radians.
        kappa: Concentration parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, kappa))


__all__ = [
    "logpdf",
    "pdf",
]
