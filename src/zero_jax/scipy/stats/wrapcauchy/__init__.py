"""Frontend API routing for jax.scipy.stats.wrapcauchy."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, cos, exp, float32, log


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wrapped Cauchy log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``c``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        c = asarray(args[1], dtype=float32)
        return log(1.0 - c * c) - math.log(2.0 * math.pi) - log(1.0 + c * c - 2.0 * c * cos(x))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wrapped Cauchy probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``c``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
