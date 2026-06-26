"""Frontend API routing for jax.scipy.ndimage."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def map_coordinates(*args: Any, **kwargs: Any) -> Any:
    """Map the input array to new coordinates using interpolation."""
    return getattr(_ops, "map_coordinates")(*args, **kwargs)
