"""Frontend API routing for jax.dtypes."""

from typing import Any


class bfloat16:
    """bfloat16 floating-point values"""

    pass


def canonicalize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
    raise NotImplementedError("canonicalize_dtype not yet implemented in zero-jax")


class extended:
    """Scalar class for extended dtypes."""

    pass


class finfo:
    """finfo(dtype)"""

    pass


float0: Any = None


class iinfo:
    """Mock implementation for iinfo."""

    pass


def issubdtype(*args: Any, **kwargs: Any) -> Any:
    """Returns True if first argument is a typecode lower/equal in type hierarchy."""
    raise NotImplementedError("issubdtype not yet implemented in zero-jax")


class prng_key:
    """Scalar class for PRNG Key dtypes."""

    pass


def result_type(*args: Any, **kwargs: Any) -> Any:
    """Convenience function to apply JAX argument dtype promotion."""
    raise NotImplementedError("result_type not yet implemented in zero-jax")


def scalar_type_of(*args: Any, **kwargs: Any) -> Any:
    """Return the scalar type associated with a JAX value."""
    raise NotImplementedError("scalar_type_of not yet implemented in zero-jax")
