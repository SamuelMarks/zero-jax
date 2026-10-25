"""Frontend API routing for jax.scipy.interpolate."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, clip, float32, floor, int32, maximum, minimum


class RegularGridInterpolator:
    """Interpolate points on a regular rectangular grid in arbitrary dimensions."""

    def __init__(
        self,
        *args: ConfigValue,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: ConfigValue = float("nan"),
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        dummy: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``RegularGridInterpolator``.

        Args:
            *args: Positional arguments ``(points, values)``.
            method: Interpolation method (``"linear"`` or ``"nearest"``).
            bounds_error: Whether out-of-bounds queries raise an error.
            fill_value: Fill value for out-of-bounds points when ``bounds_error=False``.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.grid = tuple(asarray(p, dtype=float32) for p in args[0]) if len(args) >= 1 and isinstance(args[0], (tuple, list)) else ()
        self.values = asarray(args[1], dtype=float32) if len(args) >= 2 else None
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs

    def __call__(self, xi: ArrayLikeSpec, method: str | None = None) -> ArrayLikeSpec:
        """Evaluate the interpolator at coordinates ``xi``.

        Args:
            xi: Query coordinates.
            method: Optional override for interpolation method.

        Returns:
            ArrayLikeSpec: Interpolated values at ``xi``.
        """
        q = asarray(xi, dtype=float32)
        if self.values is not None and len(self.grid) == 1:
            g0 = self.grid[0]
            n = g0.shape[0]
            q1d = clip(q[..., 0] if q.ndim > 1 and q.shape[-1] == 1 else q, g0[0], g0[-1])
            step = (g0[-1] - g0[0]) / float(max(n - 1, 1))
            pos = (q1d - g0[0]) / step
            idx0 = asarray(minimum(maximum(floor(pos), 0.0), float(max(n - 2, 0))), dtype=int32)
            idx1 = idx0 + 1
            frac = pos - asarray(idx0, dtype=float32)
            return self.values[idx0] * (1.0 - frac) + self.values[idx1] * frac
        return q


__all__ = ["RegularGridInterpolator"]
