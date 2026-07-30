"""Frontend API routing for jax.scipy.ndimage."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def map_coordinates(*args: Any, **kwargs: Any) -> Any:
    """Map the input array to new coordinates using interpolation."""
    return _ops.map_coordinates(*args, **kwargs)
