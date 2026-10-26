"""Frontend API routing for jax.scipy.interpolate."""

from __future__ import annotations

from collections.abc import Sequence

from zero_jax._types import ArrayLike
from zero_jax.numpy.lax_numpy import (
    any as _any,
)
from zero_jax.numpy.lax_numpy import (
    asarray,
    clip,
    floor,
    full_like,
    int32,
    maximum,
    minimum,
    ndarray,
    ones_like,
    where,
    zeros_like,
)
from zero_jax.numpy.lax_numpy import (
    round as _round,
)


class RegularGridInterpolator:
    """Interpolate points on a regular rectangular grid in arbitrary dimensions."""

    def __init__(
        self,
        points: Sequence[ArrayLike] = (),
        values: ArrayLike | None = None,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float | None = float("nan"),
    ) -> None:
        """Initialize ``RegularGridInterpolator``.

        Args:
            points: Tuple or sequence of 1D arrays defining the grid coordinates along each dimension.
            values: Data values on the regular grid.
            method: Interpolation method (``"linear"`` or ``"nearest"``).
            bounds_error: Whether out-of-bounds queries raise a ``ValueError``.
            fill_value: Fill value for out-of-bounds points when ``bounds_error=False`` (``None`` extrapolates/clamps).
        """
        self.grid: tuple[ndarray, ...] = tuple(asarray(p) for p in points)
        self.values: ndarray | None = asarray(values) if values is not None else None
        self.method: str = method
        self.bounds_error: bool = bounds_error
        self.fill_value: float | None = fill_value

    def __call__(self, xi: ArrayLike, method: str | None = None) -> ndarray:
        """Evaluate the interpolator at coordinates ``xi``.

        Args:
            xi: Query coordinates of shape ``(..., ndim)`` or ``(N,)`` for 1D grids.
            method: Optional override for interpolation method (``"linear"`` or ``"nearest"``).

        Returns:
            ndarray: Interpolated values at ``xi``.

        Raises:
            ValueError: If ``bounds_error=True`` and any query coordinate is outside the grid bounds,
                or if ``method`` is not ``"linear"`` or ``"nearest"``.
        """
        q = asarray(xi)
        if self.values is None or len(self.grid) == 0:
            return q
        eff_method = (method or self.method).lower()
        if eff_method not in ("linear", "nearest"):
            raise ValueError(f"Method '{eff_method}' is not defined; expected 'linear' or 'nearest'.")

        ndim = len(self.grid)
        val_dtype = self.values.dtype
        q_coords = [q[..., 0] if q.ndim > 1 and q.shape[-1] == 1 else q] if ndim == 1 else [q[..., d] for d in range(ndim)]

        oob = zeros_like(q_coords[0]) > 0.0
        pos_list: list[ndarray] = []
        for d, g in enumerate(self.grid):
            coord = asarray(q_coords[d], dtype=val_dtype)
            g_arr = asarray(g, dtype=val_dtype)
            n = g_arr.shape[0]
            g_min = g_arr[0]
            g_max = g_arr[-1]
            oob = oob | (coord < g_min) | (coord > g_max)
            coord_clamped = clip(coord, g_min, g_max)
            step = (g_max - g_min) / float(max(n - 1, 1))
            pos_list.append((coord_clamped - g_min) / step)

        if self.bounds_error and bool(_any(oob)):
            raise ValueError("One of the requested xi is out of bounds in RegularGridInterpolator.")

        if eff_method == "nearest":
            idx_tuple = tuple(asarray(clip(_round(pos_list[d]), 0.0, float(max(self.grid[d].shape[0] - 1, 0))), dtype=int32) for d in range(ndim))
            res = self.values[idx_tuple] if ndim > 1 else self.values[idx_tuple[0]]
        else:
            idx0_list: list[ndarray] = []
            idx1_list: list[ndarray] = []
            frac_list: list[ndarray] = []
            for d in range(ndim):
                n = self.grid[d].shape[0]
                i0 = asarray(minimum(maximum(floor(pos_list[d]), 0.0), float(max(n - 2, 0))), dtype=int32)
                i1 = minimum(i0 + 1, max(n - 1, 0))
                f = pos_list[d] - asarray(i0, dtype=val_dtype)
                idx0_list.append(i0)
                idx1_list.append(i1)
                frac_list.append(f)

            res = zeros_like(pos_list[0])
            for corner in range(1 << ndim):
                weight = ones_like(pos_list[0])
                corner_idx: list[ndarray] = []
                for d in range(ndim):
                    if (corner >> d) & 1:
                        corner_idx.append(idx1_list[d])
                        weight = weight * frac_list[d]
                    else:
                        corner_idx.append(idx0_list[d])
                        weight = weight * (1.0 - frac_list[d])
                corner_val = self.values[tuple(corner_idx)] if ndim > 1 else self.values[corner_idx[0]]
                res = res + corner_val * weight

        if self.fill_value is not None:
            res = where(oob, full_like(res, self.fill_value), res)
        return res


__all__ = ["RegularGridInterpolator"]
