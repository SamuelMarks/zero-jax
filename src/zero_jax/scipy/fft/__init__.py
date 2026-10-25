"""Frontend API routing for jax.scipy.fft."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import arange, asarray, cos, float32
from zero_jax.numpy.lax_numpy import sum as _sum


def dct(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the 1D discrete cosine transform (DCT-II) of the input.

    Args:
        *args: Input parameter ``args`` (``x``, ``type=2``, ``n=None``, ``axis=-1``, ``norm=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = asarray(args[0], dtype=float32)
        n = x.shape[-1]
        k = arange(0, n, dtype=float32)[:, None]
        idx = arange(0, n, dtype=float32)[None, :]
        basis = 2.0 * cos(math.pi * k * (2.0 * idx + 1.0) / (2.0 * float(n)))
        return _sum(x[..., None, :] * basis, axis=-1)
    return _ops.dct(*args, **kwargs)


def dctn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the multidimensional discrete cosine transform of the input.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return dct(*args, **kwargs)
    return _ops.dctn(*args, **kwargs)


def idct(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the 1D inverse discrete cosine transform of the input.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        x = asarray(args[0], dtype=float32)
        n = x.shape[-1]
        k = arange(0, n, dtype=float32)[None, :]
        idx = arange(0, n, dtype=float32)[:, None]
        basis = cos(math.pi * k * (2.0 * idx + 1.0) / (2.0 * float(n)))
        return (x[..., :1] + 2.0 * _sum(x[..., None, 1:] * basis[:, 1:], axis=-1)) / (2.0 * float(n))
    return _ops.idct(*args, **kwargs)


def idctn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the multidimensional inverse discrete cosine transform of the input.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return idct(*args, **kwargs)
    return _ops.idctn(*args, **kwargs)
