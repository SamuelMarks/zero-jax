"""Frontend API routing for jax.errors."""

from __future__ import annotations


class _BaseJAXException(Exception):
    """Base exception class for JAX-compatible error types."""

    def __init__(self, message: str = "") -> None:
        """Initialize the JAX exception with an error message.

        Args:
            message: Human-readable exception message.
        """
        super().__init__(message)
        self.message: str = str(message)


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

    def __init__(self, frames: list[str] | tuple[str, ...] | None = None) -> None:
        """Initialize a ``SimplifiedTraceback`` instance.

        Args:
            frames: Optional sequence of traceback frame descriptions.
        """
        self.frames: list[str] = list(frames) if frames is not None else []

    def __enter__(self) -> SimplifiedTraceback:
        """Enter the simplified traceback context.

        Returns:
            SimplifiedTraceback: Self reference.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: BaseException | None,
    ) -> None:
        """Exit the simplified traceback context without suppressing exceptions.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Traceback object if raised.
        """
        _ = (exc_type, exc_val, exc_tb)


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
