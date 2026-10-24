"""Frontend API routing for jax.errors."""

from __future__ import annotations

from zero_jax._types import ConfigValue


class _BaseJAXException(Exception):
    """Base exception class for JAX-compatible error types."""

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize the JAX exception with message arguments and optional metadata.

        Args:
            *args: Positional exception message arguments.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(*args)
        self.id = id
        self.name = name
        self.value = value


class ConcretizationTypeError(_BaseJAXException, TypeError):
    """Error raised when a JAX Tracer object is used where a concrete value is required."""


class JAXIndexError(_BaseJAXException, IndexError):
    """Error raised for invalid out-of-bounds or malformed indexing in JAX operations."""


class JAXTypeError(_BaseJAXException, TypeError):
    """Error raised for invalid argument or operand types in JAX operations."""


class KeyReuseError(_BaseJAXException, RuntimeError):
    """Error raised when a PRNG key is reused in an unsafe manner."""


class NonConcreteBooleanIndexError(_BaseJAXException, IndexError):
    """Error raised when a program attempts to use non-concrete boolean indices."""


class SimplifiedTraceback:
    """Context manager and helper for simplified JAX tracebacks."""

    def __init__(
        self,
        frames: ConfigValue = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize a ``SimplifiedTraceback`` instance.

        Args:
            frames: Optional sequence of traceback frames.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.frames = frames
        self.id = id
        self.name = name
        self.value = value


class TracerArrayConversionError(_BaseJAXException, TypeError):
    """Error raised when converting a JAX Tracer object into a standard NumPy array."""


class TracerBoolConversionError(_BaseJAXException, TypeError):
    """Error raised when a traced value in JAX is used in a Python boolean context."""


class TracerIntegerConversionError(_BaseJAXException, TypeError):
    """Error raised when a JAX Tracer object is used where a Python integer is required."""


class UnexpectedTracerError(_BaseJAXException, RuntimeError):
    """Error raised when using a JAX value that has leaked out of a traced function."""


__all__ = [
    "ConcretizationTypeError",
    "JAXIndexError",
    "JAXTypeError",
    "KeyReuseError",
    "NonConcreteBooleanIndexError",
    "SimplifiedTraceback",
    "TracerArrayConversionError",
    "TracerBoolConversionError",
    "TracerIntegerConversionError",
    "UnexpectedTracerError",
]
