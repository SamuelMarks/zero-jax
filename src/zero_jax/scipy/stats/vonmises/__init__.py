"""Frontend API routing for jax.scipy.stats.vonmises."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, cos, exp, float32, log
from zero_jax.scipy.special import i0e


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Von Mises log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``kappa``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        kappa = asarray(args[1], dtype=float32)
        return kappa * (cos(x) - 1.0) - math.log(2.0 * math.pi) - log(i0e(kappa))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Von Mises probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``kappa``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
