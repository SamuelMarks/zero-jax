"""Frontend API routing for jax.scipy.stats.multivariate_normal."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.numpy.linalg import inv, slogdet


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multivariate normal log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``mean``, ``cov``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        x = asarray(args[0], dtype=float32)
        mean = asarray(args[1], dtype=float32)
        cov = asarray(args[2], dtype=float32)
        diff = x - mean
        k = float(diff.shape[-1]) if diff.shape else 1.0
        _, logdet = slogdet(cov)
        inv_cov = inv(cov)
        quad = _sum(diff * (diff @ inv_cov), axis=-1)
        return -0.5 * (k * math.log(2.0 * math.pi) + logdet + quad)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multivariate normal probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``mean``, ``cov``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
