"""Frontend API routing for jax.util."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def as_hashable_function(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for as_hashable_function.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.as_hashable_function(*args, **kwargs)


def cache(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for cache.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.cache(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for safe_map.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.safe_map(*args, **kwargs)


def safe_zip(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for safe_zip.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.safe_zip(*args, **kwargs)


def split_dict(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_dict.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.split_dict(*args, **kwargs)


def split_list(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_list.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.split_list(*args, **kwargs)


def split_list_checked(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_list_checked.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.split_list_checked(*args, **kwargs)


def split_merge(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_merge.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.split_merge(*args, **kwargs)


def subvals(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for subvals.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.subvals(*args, **kwargs)


def toposort(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for toposort.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.toposort(*args, **kwargs)


def unzip2(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-2 tuples into two tuples."""
    return _ops.unzip2(*args, **kwargs)


def wrap_name(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for wrap_name.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.wrap_name(*args, **kwargs)


def wraps(*args: Any, **kwargs: Any) -> Any:
    """Like functools.wraps, but with finer-grained control over the name and docstring"""
    return _ops.wraps(*args, **kwargs)
