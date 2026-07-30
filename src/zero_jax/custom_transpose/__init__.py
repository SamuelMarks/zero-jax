"""Frontend API routing for jax.custom_transpose."""

from typing import Any


class custom_transpose:
    """Frontend state holder for custom_transpose."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover
