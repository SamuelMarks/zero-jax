"""Frontend API routing for jax.scipy.stats.dirichlet."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.scipy.special import gammaln, xlogy


def logpdf(x: ArrayLikeSpec, alpha: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dirichlet log probability distribution function.

    Args:
        x: Simplex probability vector(s).
        alpha: Concentration parameter vector.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    alpha_arr = asarray(alpha, dtype=float32)
    ln_b = _sum(gammaln(alpha_arr), axis=-1) - gammaln(_sum(alpha_arr, axis=-1))
    return _sum(xlogy(alpha_arr - 1.0, x_arr), axis=-1) - ln_b


def pdf(x: ArrayLikeSpec, alpha: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dirichlet probability distribution function.

    Args:
        x: Simplex probability vector(s).
        alpha: Concentration parameter vector.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, alpha))


__all__ = [
    "logpdf",
    "pdf",
]
