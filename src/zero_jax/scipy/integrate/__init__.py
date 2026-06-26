"""Frontend API routing for jax.scipy.integrate."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def trapezoid(*args: Any, **kwargs: Any) -> Any:
    """Integrate along the given axis using the composite trapezoidal rule."""
    return getattr(_ops, "trapezoid")(*args, **kwargs)
