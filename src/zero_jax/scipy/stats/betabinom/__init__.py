"""Frontend API routing for jax.scipy.stats.betabinom."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, where
from zero_jax.scipy.special import betaln


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta-binomial log probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``n``, ``a``, ``b``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 4:
        from zero_jax.scipy.special import gammaln

        k = asarray(args[0], dtype=float32) - asarray(args[4] if len(args) > 4 else kwargs.get("loc", 0.0), dtype=float32)
        n = asarray(args[1], dtype=float32)
        a = asarray(args[2], dtype=float32)
        b = asarray(args[3], dtype=float32)
        comb = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
        res = comb + betaln(k + a, n - k + b) - betaln(a, b)
        return where((k >= 0.0) & (k <= n), res, float("-inf"))
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Beta-binomial probability mass function.

    Args:
        *args: Input parameter ``args`` (``k``, ``n``, ``a``, ``b``, ``loc=0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 4:
        return exp(logpmf(*args, **kwargs))
    return _ops.pmf(*args, **kwargs)
