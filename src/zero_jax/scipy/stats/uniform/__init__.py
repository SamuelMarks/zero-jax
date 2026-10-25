"""Frontend API routing for jax.scipy.stats.uniform."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, ones_like, where, zeros_like


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Uniform log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        return where((x >= loc) & (x <= loc + scale), -log(scale) * ones_like(x), float("-inf"))
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Uniform probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Uniform cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        z = (x - loc) / scale
        return where(z < 0.0, zeros_like(z), where(z > 1.0, ones_like(z), z))
    return _ops.cdf(*args, **kwargs)


def ppf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Uniform distribution percent point function.

    Args:
        *args: Input parameter ``args`` (``q``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        q = asarray(args[0], dtype=float32)
        loc = asarray(args[1] if len(args) > 1 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[2] if len(args) > 2 else kwargs.get("scale", 1.0), dtype=float32)
        return where((q >= 0.0) & (q <= 1.0), loc + q * scale, float("nan"))
    return _ops.ppf(*args, **kwargs)
