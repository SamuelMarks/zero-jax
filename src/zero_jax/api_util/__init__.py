"""Frontend API routing for jax.api_util."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def argnums_partial(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for argnums_partial."""
    return getattr(_ops, "argnums_partial")(*args, **kwargs)


def donation_vector(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple with a boolean value for each leaf in args and kwargs."""
    return getattr(_ops, "donation_vector")(*args, **kwargs)


def flatten_axes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for flatten_axes."""
    return getattr(_ops, "flatten_axes")(*args, **kwargs)


def flatten_fun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "flatten_fun")(*args, **kwargs)


def flatten_fun_nokwargs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "flatten_fun_nokwargs")(*args, **kwargs)


def rebase_donate_argnums(*args: Any, **kwargs: Any) -> Any:
    """Shifts donate to account for static."""
    return getattr(_ops, "rebase_donate_argnums")(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_map."""
    return getattr(_ops, "safe_map")(*args, **kwargs)


def shaped_abstractify(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shaped_abstractify."""
    return getattr(_ops, "shaped_abstractify")(*args, **kwargs)
