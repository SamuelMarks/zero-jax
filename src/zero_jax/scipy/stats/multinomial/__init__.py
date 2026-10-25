"""Frontend API routing for jax.scipy.stats.multinomial."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32
from zero_jax.numpy.lax_numpy import sum as _sum
from zero_jax.scipy.special import gammaln, xlogy


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multinomial log probability mass function.

    Args:
        *args: Input parameter ``args`` (``x``, ``n``, ``p``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        x = asarray(args[0], dtype=float32)
        n = asarray(args[1], dtype=float32)
        p = asarray(args[2], dtype=float32)
        return gammaln(n + 1.0) + _sum(xlogy(x, p) - gammaln(x + 1.0), axis=-1)
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multinomial probability mass function.

    Args:
        *args: Input parameter ``args`` (``x``, ``n``, ``p``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)
