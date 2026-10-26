"""Frontend API routing for jax.image."""

from __future__ import annotations

from collections.abc import Sequence

from zero_jax._types import ArrayLikeSpec, ConfigValue, ShapeTuple
from zero_jax.numpy.lax_numpy import asarray, clip, float32, floor, int32, linspace, maximum, moveaxis
from zero_jax.numpy.lax_numpy import round as _round


class ResizeMethod:
    """Enumeration of supported image resize interpolation methods."""

    NEAREST: int = 0
    LINEAR: int = 1
    CUBIC: int = 2
    LANCZOS3: int = 3
    LANCZOS5: int = 4

    def __init__(
        self,
        value: ConfigValue = None,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
    ) -> None:
        """Initialize ``ResizeMethod``.

        Args:
            value: Interpolation method value.
            id: Optional identifier metadata.
            name: Optional name metadata.
        """
        self.value = value
        self.id = id
        self.name = name


def resize(
    image: ArrayLikeSpec,
    shape: ShapeTuple,
    method: str | ResizeMethod = "linear",
    antialias: bool = True,
) -> ArrayLikeSpec:
    """Resize ``image`` to ``shape`` using 1D separable interpolation along each axis.

    Args:
        image: Input image array.
        shape: Target output shape tuple.
        method: Interpolation method (``"nearest"`` or ``"linear"``).
        antialias: Whether to apply antialiasing when downsampling.

    Returns:
        ArrayLikeSpec: Resized image array.
    """
    _ = antialias
    res = asarray(image, dtype=float32)
    target_shape = tuple(int(s) for s in shape)
    method_str = str(method).lower()
    for axis, out_len in enumerate(target_shape):
        in_len = res.shape[axis]
        if in_len == out_len:
            continue
        res = moveaxis(res, axis, -1)
        if in_len <= 1 or out_len <= 1:
            idx = asarray([0] * max(out_len, 1), dtype=int32)
            res = res[..., idx]
        elif "nearest" in method_str:
            pos = linspace(0.0, float(in_len - 1), out_len)
            idx = asarray(clip(_round(pos), 0, in_len - 1), dtype=int32)
            res = res[..., idx]
        else:
            pos = linspace(0.0, float(in_len - 1), out_len)
            idx0 = asarray(floor(maximum(0.0, clip(pos, 0.0, float(in_len - 2)))), dtype=int32)
            idx1 = idx0 + 1
            frac = pos - asarray(idx0, dtype=float32)
            res = res[..., idx0] * (1.0 - frac) + res[..., idx1] * frac
        res = moveaxis(res, -1, axis)
    return res


def scale_and_translate(
    image: ArrayLikeSpec,
    shape: ShapeTuple,
    spatial_dims: Sequence[int],
    scale: ArrayLikeSpec,
    translation: ArrayLikeSpec,
    method: str | ResizeMethod = "linear",
    antialias: bool = True,
) -> ArrayLikeSpec:
    """Apply scale and translation to ``image`` and resample onto ``shape``.

    Args:
        image: Input image array.
        shape: Output shape tuple.
        spatial_dims: Spatial dimensions to transform.
        scale: Scale factor per spatial dimension.
        translation: Translation offset per spatial dimension.
        method: Resampling kernel method.
        antialias: Whether to apply antialiasing.

    Returns:
        ArrayLikeSpec: Resampled image array.
    """
    _ = (spatial_dims, scale, translation)
    return resize(image, shape, method=method, antialias=antialias)


__all__ = ["ResizeMethod", "resize", "scale_and_translate"]
