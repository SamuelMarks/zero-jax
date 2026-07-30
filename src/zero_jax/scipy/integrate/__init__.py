"""Frontend API routing for jax.scipy.integrate."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def trapezoid(*args: Any, **kwargs: Any) -> Any:
    """Integrate along the given axis using the composite trapezoidal rule."""
    return _ops.trapezoid(*args, **kwargs)
