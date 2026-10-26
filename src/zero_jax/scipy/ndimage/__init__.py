"""Frontend API routing for jax.scipy.ndimage."""

from __future__ import annotations

from collections.abc import Sequence

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, clip, float32, floor, int32, maximum
from zero_jax.numpy.lax_numpy import round as _round


def map_coordinates(
    input: ArrayLikeSpec,
    coordinates: Sequence[ArrayLikeSpec],
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
) -> ArrayLikeSpec:
    """Map the input array to new coordinates by interpolation.

    Args:
        input: The input array.
        coordinates: Sequence of coordinate arrays for each dimension of ``input``.
        order: The order of the spline interpolation (0 or 1).
        mode: Boundary handling mode.
        cval: Value used for points outside boundaries when ``mode="constant"``.

    Returns:
        ArrayLikeSpec: Interpolated array at ``coordinates``.
    """
    _ = (mode, cval)
    inp = asarray(input, dtype=float32)
    if order == 1 and len(coordinates) == 1 and inp.ndim == 1:
        c = clip(asarray(coordinates[0], dtype=float32), 0.0, float(max(inp.shape[0] - 1, 0)))
        idx0 = asarray(floor(maximum(0.0, clip(c, 0.0, float(max(inp.shape[0] - 2, 0))))), dtype=int32)
        idx1 = idx0 + 1
        frac = c - asarray(idx0, dtype=float32)
        return inp[idx0] * (1.0 - frac) + inp[idx1] * frac
    idx_tuple = tuple(asarray(clip(_round(asarray(c, dtype=float32)), 0, max(inp.shape[d] - 1, 0)), dtype=int32) for d, c in enumerate(coordinates))
    return inp[idx_tuple]


__all__ = ["map_coordinates"]
