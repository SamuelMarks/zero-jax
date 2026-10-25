"""Frontend API routing for jax.scipy.ndimage."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, clip, float32, int32
from zero_jax.numpy.lax_numpy import round as _round


def map_coordinates(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Map the input array to new coordinates using interpolation.

    Args:
        *args: Input parameter ``args`` (``input``, ``coordinates``, ``order``, ``mode="constant"``, ``cval=0.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        inp = asarray(args[0], dtype=float32)
        coords = list(args[1]) if isinstance(args[1], (list, tuple)) else [args[1]]
        idx_tuple = tuple(asarray(clip(_round(asarray(c, dtype=float32)), 0, max(inp.shape[d] - 1, 0)), dtype=int32) for d, c in enumerate(coords))
        return inp[idx_tuple]
    return _ops.map_coordinates(*args, **kwargs)
