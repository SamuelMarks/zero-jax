"""Frontend API routing for jax.scipy.fft."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec, AxisSpec, ShapeTuple
from zero_jax.numpy.lax_numpy import arange, asarray, cos, float32, moveaxis
from zero_jax.numpy.lax_numpy import sum as _sum


def dct(
    x: ArrayLikeSpec,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> ArrayLikeSpec:
    """Compute the 1-D Discrete Cosine Transform (DCT-II).

    Args:
        x: Input array.
        type: Type of the DCT (only type 2 supported).
        n: Optional length of the transform.
        axis: Axis along which the dct is computed.
        norm: Normalization mode (``None`` or ``"ortho"``).

    Returns:
        ArrayLikeSpec: Transformed array.
    """
    _ = (type, n)
    arr = moveaxis(asarray(x, dtype=float32), axis, -1)
    length = arr.shape[-1]
    k = arange(length, dtype=float32)[:, None]
    idx = arange(length, dtype=float32)[None, :]
    basis = cos(math.pi * (idx + 0.5) * k / float(max(length, 1)))
    res = 2.0 * _sum(arr[..., None, :] * basis, axis=-1)
    if norm == "ortho":
        scale = asarray([1.0 / math.sqrt(4.0 * max(length, 1))] + [1.0 / math.sqrt(2.0 * max(length, 1))] * max(length - 1, 0), dtype=float32)
        res = res * scale
    return moveaxis(res, -1, axis)


def dctn(
    x: ArrayLikeSpec,
    type: int = 2,
    s: ShapeTuple | None = None,
    axes: AxisSpec = None,
    norm: str | None = None,
) -> ArrayLikeSpec:
    """Compute the multidimensional Discrete Cosine Transform along specified axes.

    Args:
        x: Input array.
        type: Type of the DCT.
        s: Optional shape of the result.
        axes: Axes over which to compute the DCT.
        norm: Normalization mode.

    Returns:
        ArrayLikeSpec: Transformed array.
    """
    _ = s
    res = asarray(x, dtype=float32)
    ax_tuple = tuple(axes) if isinstance(axes, (list, tuple)) else tuple(range(res.ndim))
    for ax in ax_tuple:
        res = dct(res, type=type, axis=ax, norm=norm)
    return res


def idct(
    x: ArrayLikeSpec,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> ArrayLikeSpec:
    """Compute the 1-D Inverse Discrete Cosine Transform.

    Args:
        x: Input array.
        type: Type of the IDCT.
        n: Optional length of the transform.
        axis: Axis along which the idct is computed.
        norm: Normalization mode.

    Returns:
        ArrayLikeSpec: Inverse transformed array.
    """
    _ = (type, n, norm)
    arr = moveaxis(asarray(x, dtype=float32), axis, -1)
    length = arr.shape[-1]
    k = arange(length, dtype=float32)[None, :]
    idx = arange(length, dtype=float32)[:, None]
    basis = cos(math.pi * (idx + 0.5) * k / float(max(length, 1)))
    weights = asarray([0.5] + [1.0] * max(length - 1, 0), dtype=float32)
    res = _sum(arr[..., None, :] * weights * basis, axis=-1) / float(max(length, 1))
    return moveaxis(res, -1, axis)


def idctn(
    x: ArrayLikeSpec,
    type: int = 2,
    s: ShapeTuple | None = None,
    axes: AxisSpec = None,
    norm: str | None = None,
) -> ArrayLikeSpec:
    """Compute the multidimensional Inverse Discrete Cosine Transform.

    Args:
        x: Input array.
        type: Type of the IDCT.
        s: Optional shape of the result.
        axes: Axes over which to compute the IDCT.
        norm: Normalization mode.

    Returns:
        ArrayLikeSpec: Inverse transformed array.
    """
    _ = s
    res = asarray(x, dtype=float32)
    ax_tuple = tuple(axes) if isinstance(axes, (list, tuple)) else tuple(range(res.ndim))
    for ax in ax_tuple:
        res = idct(res, type=type, axis=ax, norm=norm)
    return res


__all__ = ["dct", "dctn", "idct", "idctn"]
