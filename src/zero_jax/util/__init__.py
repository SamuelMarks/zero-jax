"""Frontend API routing for jax.util."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def as_hashable_function(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for as_hashable_function.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.as_hashable_function(*args, **kwargs)


def cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for cache.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.cache(*args, **kwargs)


def safe_map(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_map.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.safe_map(*args, **kwargs)


def safe_zip(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_zip.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.safe_zip(*args, **kwargs)


def split_dict(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_dict.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.split_dict(*args, **kwargs)


def split_list(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.split_list(*args, **kwargs)


def split_list_checked(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list_checked.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.split_list_checked(*args, **kwargs)


def split_merge(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_merge.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.split_merge(*args, **kwargs)


def subvals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for subvals.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.subvals(*args, **kwargs)


def toposort(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for toposort.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.toposort(*args, **kwargs)


def unzip2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Unzip sequence of length-2 tuples into two tuples.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.unzip2(*args, **kwargs)


def wrap_name(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for wrap_name.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.wrap_name(*args, **kwargs)


def wraps(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Like functools.wraps, but with finer-grained control over the name and docstring.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.wraps(*args, **kwargs)
