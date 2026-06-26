"""Frontend API routing for jax.util."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    pass


def as_hashable_function(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for as_hashable_function."""
    return getattr(_ops, "as_hashable_function")(*args, **kwargs)


def cache(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for cache."""
    return getattr(_ops, "cache")(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_map."""
    return getattr(_ops, "safe_map")(*args, **kwargs)


def safe_zip(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_zip."""
    return getattr(_ops, "safe_zip")(*args, **kwargs)


def split_dict(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_dict."""
    return getattr(_ops, "split_dict")(*args, **kwargs)


def split_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_list."""
    return getattr(_ops, "split_list")(*args, **kwargs)


def split_list_checked(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_list_checked."""
    return getattr(_ops, "split_list_checked")(*args, **kwargs)


def split_merge(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_merge."""
    return getattr(_ops, "split_merge")(*args, **kwargs)


def subvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subvals."""
    return getattr(_ops, "subvals")(*args, **kwargs)


def toposort(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for toposort."""
    return getattr(_ops, "toposort")(*args, **kwargs)


def unzip2(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-2 tuples into two tuples."""
    return getattr(_ops, "unzip2")(*args, **kwargs)


def wrap_name(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for wrap_name."""
    return getattr(_ops, "wrap_name")(*args, **kwargs)


def wraps(*args: Any, **kwargs: Any) -> Any:
    """Like functools.wraps, but with finer-grained control over the name and docstring"""
    return getattr(_ops, "wraps")(*args, **kwargs)
