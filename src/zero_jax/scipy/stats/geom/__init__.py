"""Frontend API routing for jax.scipy.stats.geom."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, log1p, where


def logpmf(k: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Geometric log probability mass function.

    Args:
        k: Number of trials until first success.
        p: Success probability per trial.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability mass.
    """
    k_arr = asarray(k, dtype=float32) - asarray(loc, dtype=float32)
    p_arr = asarray(p, dtype=float32)
    val = (k_arr - 1.0) * log1p(-p_arr) + log(p_arr)
    return where(k_arr >= 1.0, val, float("-inf"))


def pmf(k: ArrayLikeSpec, p: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0) -> ArrayLikeSpec:
    """Geometric probability mass function.

    Args:
        k: Number of trials until first success.
        p: Success probability per trial.
        loc: Location offset parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability mass.
    """
    return exp(logpmf(k, p, loc))


__all__ = [
    "logpmf",
    "pmf",
]
