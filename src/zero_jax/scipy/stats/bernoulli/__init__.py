"""Frontend API routing for jax.scipy.stats.bernoulli."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, floor, ones_like, where, zeros_like
from zero_jax.scipy.special import xlog1py, xlogy


def logpmf(k: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Bernoulli log probability mass function.

    Args:
        k: Quantiles (0 or 1).
        p: Success probability parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    val = xlogy(k_arr, p_arr) + xlog1py(1.0 - k_arr, -p_arr)
    return where((k_arr == 0.0) | (k_arr == 1.0), val, float("-inf"))


def pmf(k: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Bernoulli probability mass function.

    Args:
        k: Quantiles (0 or 1).
        p: Success probability parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, p, loc))


def cdf(k: ArrayLikeSpec, p: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli cumulative distribution function.

    Args:
        k: Quantiles.
        p: Success probability parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    k_arr = floor(asarray(k, dtype=float32))
    p_arr = asarray(p, dtype=float32)
    return where(k_arr < 0.0, zeros_like(k_arr), where(k_arr >= 1.0, ones_like(k_arr), 1.0 - p_arr))


def ppf(q: ArrayLikeSpec, p: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli percent point function.

    Args:
        q: Lower tail probability.
        p: Success probability parameter.

    Returns:
        ArrayLikeSpec: Quantile corresponding to ``q``.
    """
    q_arr = asarray(q, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    return where((q_arr < 0.0) | (q_arr > 1.0), float("nan"), where(q_arr <= 1.0 - p_arr, zeros_like(q_arr), ones_like(q_arr)))


__all__ = [
    "cdf",
    "logpmf",
    "pmf",
    "ppf",
]
