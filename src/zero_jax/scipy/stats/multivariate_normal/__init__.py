"""Frontend API routing for jax.scipy.stats.multivariate_normal."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.numpy.linalg import inv, slogdet


def logpdf(x: ArrayLikeSpec, mean: ArrayLikeSpec, cov: ArrayLikeSpec, allow_singular: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Multivariate normal log probability distribution function.

    Args:
        x: Input quantiles of shape ``(..., k)``.
        mean: Mean vector of shape ``(..., k)``.
        cov: Covariance matrix of shape ``(..., k, k)``.
        allow_singular: Unused singular covariance flag.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    _ = allow_singular
    x_arr = asarray(x, dtype=float32)
    mean_arr = asarray(mean, dtype=float32)
    cov_arr = asarray(cov, dtype=float32)
    diff = x_arr - mean_arr
    k = float(diff.shape[-1]) if diff.shape else 1.0
    _, logdet = slogdet(cov_arr)
    inv_cov = inv(cov_arr)
    quad = _sum(diff * (diff @ inv_cov), axis=-1)
    return -0.5 * (k * math.log(2.0 * math.pi) + logdet + quad)


def pdf(x: ArrayLikeSpec, mean: ArrayLikeSpec, cov: ArrayLikeSpec, allow_singular: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Multivariate normal probability distribution function.

    Args:
        x: Input quantiles of shape ``(..., k)``.
        mean: Mean vector of shape ``(..., k)``.
        cov: Covariance matrix of shape ``(..., k, k)``.
        allow_singular: Unused singular covariance flag.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, mean, cov, allow_singular=allow_singular))


__all__ = [
    "logpdf",
    "pdf",
]
