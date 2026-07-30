"""Frontend API routing for jax.stages."""

from typing import Any


class ArgInfo:
    """ArgInfo(_aval: 'core.AbstractValue', donated: 'bool')"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Compiled:
    """Compiled representation of a function specialized to types/values."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CompilerOptions:
    """dict() -> new empty dictionary"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Lowered:
    """Lowering of a function specialized to argument types and values."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OutInfo:
    """OutInfo(shape: 'tuple[int, ...]', dtype: 'jax.typing.DTypeLike', sharding: 'jax.sharding.Sharding | None' = None)"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Traced:
    """Frontend state holder for Traced."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Wrapped:
    """A function ready to be traced, lowered, and compiled."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover
