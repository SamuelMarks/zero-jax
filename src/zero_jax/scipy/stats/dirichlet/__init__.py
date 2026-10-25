"""Frontend API routing for jax.scipy.stats.dirichlet."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.scipy.special import gammaln, xlogy


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dirichlet log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``alpha``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        alpha = asarray(args[1], dtype=float32)
        ln_b = _sum(gammaln(alpha), axis=-1) - gammaln(_sum(alpha, axis=-1))
        return _sum(xlogy(alpha - 1.0, x), axis=-1) - ln_b
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dirichlet probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``alpha``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
