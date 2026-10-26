"""Frontend API routing for jax.scipy.stats.binom."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, where
from zero_jax.scipy.special import gammaln, xlog1py, xlogy


def logpmf(k: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Binomial log probability mass function.

    Args:
        k: Number of successes.
        n: Number of trials.
        p: Success probability per trial.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    n_arr = asarray(n, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    comb = gammaln(n_arr + 1.0) - gammaln(k_arr + 1.0) - gammaln(n_arr - k_arr + 1.0)
    val = comb + xlogy(k_arr, p_arr) + xlog1py(n_arr - k_arr, -p_arr)
    return where((k_arr >= 0.0) & (k_arr <= n_arr), val, float("-inf"))


def pmf(k: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Binomial probability mass function.

    Args:
        k: Number of successes.
        n: Number of trials.
        p: Success probability per trial.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, n, p, loc))


__all__ = [
    "logpmf",
    "pmf",
]
