"""Frontend API routing for jax.errors."""

from typing import Any


class ConcretizationTypeError:
    """This error occurs when a JAX Tracer object is used in a context where a"""

    pass


class JAXIndexError:
    """Mock implementation for JAXIndexError."""

    pass


class JAXTypeError:
    """Mock implementation for JAXTypeError."""

    pass


class KeyReuseError:
    """This error occurs when a PRNG key is reused in an unsafe manner."""

    pass


class NonConcreteBooleanIndexError:
    """This error occurs when a program attempts to use non-concrete boolean indices"""

    pass


class SimplifiedTraceback:
    """Mock implementation for SimplifiedTraceback."""

    pass


class TracerArrayConversionError:
    """This error occurs when a program attempts to convert a JAX Tracer object into"""

    pass


class TracerBoolConversionError:
    """This error occurs when a traced value in JAX is used in a context where a"""

    pass


class TracerIntegerConversionError:
    """This error can occur when a JAX Tracer object is used in a context where a"""

    pass


class UnexpectedTracerError:
    """This error occurs when you use a JAX value that has leaked out of a function."""

    pass
