"""Frontend API routing for jax.scipy.integrate."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, diff, float32
from zero_jax.numpy.lax_numpy import sum as _sum


def trapezoid(y: ArrayLikeSpec, x: ArrayLikeSpec = None, dx: float = 1.0, axis: int = -1) -> ArrayLikeSpec:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y: Input array to integrate.
        x: Optional sample points corresponding to ``y`` values.
        dx: Spacing between sample points when ``x`` is ``None``.
        axis: Axis along which to integrate.

    Returns:
        ArrayLikeSpec: Definite integral approximated by trapezoidal rule.
    """
    y_arr = asarray(y, dtype=float32)
    avg_y = 0.5 * (y_arr[..., :-1] + y_arr[..., 1:])
    if x is not None:
        d = diff(asarray(x, dtype=float32), axis=axis)
        return _sum(avg_y * d, axis=axis)
    return _sum(avg_y * dx, axis=axis)


__all__ = ["trapezoid"]
