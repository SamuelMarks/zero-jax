"""Frontend API routing for jax.errors."""

from typing import Any


class ConcretizationTypeError(Exception):
    """This error occurs when a JAX Tracer object is used in a context where a"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class JAXIndexError(Exception):
    """JAX index error."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class JAXTypeError(Exception):
    """JAX type error."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class KeyReuseError(Exception):
    """This error occurs when a PRNG key is reused in an unsafe manner."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class NonConcreteBooleanIndexError(Exception):
    """This error occurs when a program attempts to use non-concrete boolean indices"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class SimplifiedTraceback:
    """Simplified traceback."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class TracerArrayConversionError(Exception):
    """This error occurs when a program attempts to convert a JAX Tracer object into"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class TracerBoolConversionError(Exception):
    """This error occurs when a traced value in JAX is used in a context where a"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class TracerIntegerConversionError(Exception):
    """This error can occur when a JAX Tracer object is used in a context where a"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class UnexpectedTracerError(Exception):
    """This error occurs when you use a JAX value that has leaked out of a function."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover
