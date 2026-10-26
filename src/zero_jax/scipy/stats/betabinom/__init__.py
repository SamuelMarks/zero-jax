"""Frontend API routing for jax.scipy.stats.betabinom."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, where
from zero_jax.scipy.special import betaln, gammaln


def logpmf(k: ArrayLikeSpec, n: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Beta-binomial log probability mass function.

    Args:
        k: Number of successes.
        n: Number of trials.
        a: First beta shape parameter.
        b: Second beta shape parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    n_arr = asarray(n, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    comb = gammaln(n_arr + 1.0) - gammaln(k_arr + 1.0) - gammaln(n_arr - k_arr + 1.0)
    res = comb + betaln(k_arr + a_arr, n_arr - k_arr + b_arr) - betaln(a_arr, b_arr)
    return where((k_arr >= 0.0) & (k_arr <= n_arr), res, float("-inf"))


def pmf(k: ArrayLikeSpec, n: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Beta-binomial probability mass function.

    Args:
        k: Number of successes.
        n: Number of trials.
        a: First beta shape parameter.
        b: Second beta shape parameter.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, n, a, b, loc))


__all__ = [
    "logpmf",
    "pmf",
]
