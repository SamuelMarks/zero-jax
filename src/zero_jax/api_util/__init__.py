"""Frontend API routing for jax.api_util."""

from typing import Any


def argnums_partial(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for argnums_partial."""
    raise NotImplementedError("argnums_partial not yet implemented in zero-jax")


def donation_vector(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple with a boolean value for each leaf in args and kwargs."""
    raise NotImplementedError("donation_vector not yet implemented in zero-jax")


def flatten_axes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for flatten_axes."""
    raise NotImplementedError("flatten_axes not yet implemented in zero-jax")


def flatten_fun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("flatten_fun not yet implemented in zero-jax")


def flatten_fun_nokwargs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("flatten_fun_nokwargs not yet implemented in zero-jax")


def rebase_donate_argnums(*args: Any, **kwargs: Any) -> Any:
    """Shifts donate to account for static."""
    raise NotImplementedError("rebase_donate_argnums not yet implemented in zero-jax")


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_map."""
    raise NotImplementedError("safe_map not yet implemented in zero-jax")


def shaped_abstractify(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shaped_abstractify."""
    raise NotImplementedError("shaped_abstractify not yet implemented in zero-jax")
