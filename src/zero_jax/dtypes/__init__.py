"""Frontend API routing for jax.dtypes."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class bfloat16:
    """bfloat16 floating-point values"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def canonicalize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
    return _ops.canonicalize_dtype(*args, **kwargs)


class extended:
    """Scalar class for extended dtypes."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class finfo:
    """finfo(dtype)"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def float0(*args: Any, **kwargs: Any) -> Any:
    """float0 type."""
    return None


class iinfo:
    """Frontend state holder for iinfo."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def issubdtype(*args: Any, **kwargs: Any) -> Any:
    """Returns True if first argument is a typecode lower/equal in type hierarchy."""
    return _ops.issubdtype(*args, **kwargs)


class prng_key:
    """Scalar class for PRNG Key dtypes."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def result_type(*args: Any, **kwargs: Any) -> Any:
    """Convenience function to apply JAX argument dtype promotion."""
    return _ops.result_type(*args, **kwargs)


def scalar_type_of(*args: Any, **kwargs: Any) -> Any:
    """Return the scalar type associated with a JAX value."""
    return _ops.scalar_type_of(*args, **kwargs)
