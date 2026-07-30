"""Frontend API routing for jax.scipy.interpolate."""

from typing import Any


class RegularGridInterpolator:
    """Interpolate points on a regular rectangular grid."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover
