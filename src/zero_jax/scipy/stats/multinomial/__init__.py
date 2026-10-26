"""Frontend API routing for jax.scipy.stats.multinomial."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.scipy.special import gammaln, xlogy


def logpmf(x: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multinomial log probability mass function.

    Args:
        x: Category count vector.
        n: Total number of trials.
        p: Category probability vector.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    x_arr = asarray(x, dtype=float32)
    n_arr = asarray(n, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    return gammaln(n_arr + 1.0) + _sum(xlogy(x_arr, p_arr) - gammaln(x_arr + 1.0), axis=-1)


def pmf(x: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multinomial probability mass function.

    Args:
        x: Category count vector.
        n: Total number of trials.
        p: Category probability vector.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(x, n, p))


__all__ = [
    "logpmf",
    "pmf",
]
