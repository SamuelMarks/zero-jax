"""Frontend API routing for jax.api_util."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def argnums_partial(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for argnums_partial.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.argnums_partial(*args, **kwargs)


def donation_vector(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple with a boolean value for each leaf in args and kwargs."""
    return _ops.donation_vector(*args, **kwargs)


def flatten_axes(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for flatten_axes.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.flatten_axes(*args, **kwargs)


def flatten_fun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.flatten_fun(*args, **kwargs)


def flatten_fun_nokwargs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.flatten_fun_nokwargs(*args, **kwargs)


def rebase_donate_argnums(*args: Any, **kwargs: Any) -> Any:
    """Shifts donate to account for static."""
    return _ops.rebase_donate_argnums(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for safe_map.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.safe_map(*args, **kwargs)


def shaped_abstractify(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for shaped_abstractify.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.shaped_abstractify(*args, **kwargs)
