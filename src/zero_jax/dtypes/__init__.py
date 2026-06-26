"""Frontend API routing for jax.dtypes."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class bfloat16:
    """bfloat16 floating-point values"""

    pass


def canonicalize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
    return getattr(_ops, "canonicalize_dtype")(*args, **kwargs)


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
    return getattr(_ops, "issubdtype")(*args, **kwargs)


class prng_key:
    """Scalar class for PRNG Key dtypes."""

    pass


def result_type(*args: Any, **kwargs: Any) -> Any:
    """Convenience function to apply JAX argument dtype promotion."""
    return getattr(_ops, "result_type")(*args, **kwargs)


def scalar_type_of(*args: Any, **kwargs: Any) -> Any:
    """Return the scalar type associated with a JAX value."""
    return getattr(_ops, "scalar_type_of")(*args, **kwargs)
