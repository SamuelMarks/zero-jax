"""Frontend API routing for jax.stages."""

from typing import Any


class ArgInfo:
    """ArgInfo(_aval: 'core.AbstractValue', donated: 'bool')"""

    pass


class Compiled:
    """Compiled representation of a function specialized to types/values."""

    pass


class CompilerOptions:
    """dict() -> new empty dictionary"""

    pass


class Lowered:
    """Lowering of a function specialized to argument types and values."""

    pass


class OutInfo:
    """OutInfo(shape: 'tuple[int, ...]', dtype: 'jax.typing.DTypeLike', sharding: 'jax.sharding.Sharding | None' = None)"""

    pass


class Traced:
    """Mock implementation for Traced."""

    pass


class Wrapped:
    """A function ready to be traced, lowered, and compiled."""

    pass
