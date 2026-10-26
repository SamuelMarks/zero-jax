"""Frontend API routing for jax.scipy.stats.nbinom."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, where
from zero_jax.scipy.special import gammaln, xlog1py, xlogy


def logpmf(k: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Negative-binomial log probability mass function.

    Args:
        k: Number of failures before ``n`` successes.
        n: Number of successes.
        p: Probability of single success.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    n_arr = asarray(n, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    comb = gammaln(k_arr + n_arr) - gammaln(n_arr) - gammaln(k_arr + 1.0)
    val = comb + xlogy(n_arr, p_arr) + xlog1py(k_arr, -p_arr)
    return where(k_arr >= 0.0, val, float("-inf"))


def pmf(k: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Negative-binomial probability mass function.

    Args:
        k: Number of failures before ``n`` successes.
        n: Number of successes.
        p: Probability of single success.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, n, p, loc))


__all__ = [
    "logpmf",
    "pmf",
]
