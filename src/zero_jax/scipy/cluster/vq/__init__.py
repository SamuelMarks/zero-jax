"""Frontend API routing for jax.scipy.cluster.vq."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import argmin, asarray, float32, sqrt
from zero_jax.numpy.lax_numpy import min as _min
from zero_jax.numpy.lax_numpy import sum as _sum


def vq(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Assign codes from a code book to a set of observations.

    Args:
        *args: Input parameter ``args`` (``obs``, ``code_book``, ``check_finite=True``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result ``(code, dist)``.
    """
    if len(args) >= 2:
        obs = asarray(args[0], dtype=float32)
        code_book = asarray(args[1], dtype=float32)
        diff = obs[:, None, :] - code_book[None, :, :]
        dists = sqrt(_sum(diff * diff, axis=-1))
        codes = argmin(dists, axis=-1)
        min_dists = _min(dists, axis=-1)
        return codes, min_dists
    return _ops.vq(*args, **kwargs)
