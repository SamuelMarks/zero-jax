"""Frontend API routing for jax.scipy.stats.poisson."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, floor, where, zeros_like
from zero_jax.scipy.special import gammaincc, gammaln, xlogy


def logpmf(k: ArrayLikeSpec, mu: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Poisson log probability mass function.

    Args:
        k: Event count quantiles.
        mu: Expected rate parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    mu_arr = asarray(mu, dtype=float32)
    val = xlogy(k_arr, mu_arr) - gammaln(k_arr + 1.0) - mu_arr
    return where(k_arr >= 0.0, val, float("-inf"))


def pmf(k: ArrayLikeSpec, mu: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Poisson probability mass function.

    Args:
        k: Event count quantiles.
        mu: Expected rate parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, mu, loc))


def cdf(k: ArrayLikeSpec, mu: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Poisson cumulative distribution function.

    Args:
        k: Event count quantiles.
        mu: Expected rate parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    k_arr = floor(asarray(k, dtype=float32) - asarray(loc, dtype=float32))
    mu_arr = asarray(mu, dtype=float32)
    return where(k_arr < 0.0, zeros_like(k_arr), gammaincc(k_arr + 1.0, mu_arr))


__all__ = [
    "cdf",
    "logpmf",
    "pmf",
]
