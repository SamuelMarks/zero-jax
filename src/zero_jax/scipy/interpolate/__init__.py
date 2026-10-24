"""Frontend API routing for jax.scipy.interpolate."""

from __future__ import annotations

from zero_jax._types import ConfigValue


class RegularGridInterpolator:
    """Interpolate points on a regular rectangular grid."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs
