"""Frontend API routing for jax.scipy.stats.nbinom."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, where
from zero_jax.scipy.special import gammaln, xlog1py, xlogy


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Negative-binomial log probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``n``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        k = asarray(args[0], dtype=float32) - asarray(args[3] if len(args) > 3 else kwargs.get("loc", 0.0), dtype=float32)
        n = asarray(args[1], dtype=float32)
        p = asarray(args[2], dtype=float32)
        comb = gammaln(k + n) - gammaln(n) - gammaln(k + 1.0)
        val = comb + xlogy(n, p) + xlog1py(k, -p)
        return where(k >= 0.0, val, float("-inf"))
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Negative-binomial probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``n``, ``p``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)
