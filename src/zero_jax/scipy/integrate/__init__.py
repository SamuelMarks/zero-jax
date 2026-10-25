"""Frontend API routing for jax.scipy.integrate."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, diff, float32
from zero_jax.numpy.lax_numpy import sum as _sum


def trapezoid(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        *args: Input parameter ``args`` (``y``, ``x=None``, ``dx=1.0``, ``axis=-1``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        y = asarray(args[0], dtype=float32)
        x = args[1] if len(args) > 1 else kwargs.get("x")
        dx = float(args[2] if len(args) > 2 else kwargs.get("dx", 1.0))
        axis = int(args[3] if len(args) > 3 else kwargs.get("axis", -1))
        avg_y = 0.5 * (y[..., :-1] + y[..., 1:])
        if x is not None:
            d = diff(asarray(x, dtype=float32), axis=axis)
            return _sum(avg_y * d, axis=axis)
        return _sum(avg_y * dx, axis=axis)
    return _ops.trapezoid(*args, **kwargs)
