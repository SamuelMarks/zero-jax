"""Frontend API routing for jax.interpreters.traceback_util.util."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class Generic:
    """Abstract base class for generic types."""

    pass


class Hashable:
    """Mock implementation for Hashable."""

    pass


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    pass


class HashablePartial:
    """Mock implementation for HashablePartial."""

    pass


class HashableWrapper:
    """Mock implementation for HashableWrapper."""

    pass


class Iterable:
    """Mock implementation for Iterable."""

    pass


class Iterator:
    """Mock implementation for Iterator."""

    pass


class NumpyComplexWarning:
    """The warning raised when casting a complex dtype to a real dtype."""

    pass


class OrderedSet:
    """Mock implementation for OrderedSet."""

    pass


class Seq:
    """All the operations on a read-only sequence."""

    pass


class StrictABC:
    """Mock implementation for StrictABC."""

    pass


class StrictABCMeta:
    """A variant of `abc.ABCMeta` which does not allow virtual subclasses."""

    pass


T: Any = None

T1: Any = None

T2: Any = None

T3: Any = None

TYPE_CHECKING: Any = None


class TypeVar:
    """Type variable."""

    pass


class Unhashable:
    """Mock implementation for Unhashable."""

    pass


class WrapKwArgs:
    """Mock implementation for WrapKwArgs."""

    pass


from . import abc

annotations: Any = None


def as_hashable_function(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for as_hashable_function."""
    return getattr(_ops, "as_hashable_function")(*args, **kwargs)


def assert_unreachable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for assert_unreachable."""
    return getattr(_ops, "assert_unreachable")(*args, **kwargs)


def cache(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for cache."""
    return getattr(_ops, "cache")(*args, **kwargs)


cache_clearing_funs: Any = None


def canonicalize_axis(*args: Any, **kwargs: Any) -> Any:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    return getattr(_ops, "canonicalize_axis")(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return getattr(_ops, "cast")(*args, **kwargs)


def ceil_of_ratio(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ceil_of_ratio."""
    return getattr(_ops, "ceil_of_ratio")(*args, **kwargs)


def check_toposort(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_toposort."""
    return getattr(_ops, "check_toposort")(*args, **kwargs)


def clear_all_caches(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for clear_all_caches."""
    return getattr(_ops, "clear_all_caches")(*args, **kwargs)


def clear_all_weakref_lru_caches(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for clear_all_weakref_lru_caches."""
    return getattr(_ops, "clear_all_weakref_lru_caches")(*args, **kwargs)


def concatenate(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    return getattr(_ops, "concatenate")(*args, **kwargs)


from . import config


def curry(*args: Any, **kwargs: Any) -> Any:
    """Curries arguments of f, returning a function on any remaining arguments."""
    return getattr(_ops, "curry")(*args, **kwargs)


from . import dataclasses


def distributed_debug_log(*args: Any, **kwargs: Any) -> Any:
    """Format and log `pairs` if config.jax_distributed_debug is enabled."""
    return getattr(_ops, "distributed_debug_log")(*args, **kwargs)


def flatten(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    return getattr(_ops, "flatten")(*args, **kwargs)


def fun_name(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for fun_name."""
    return getattr(_ops, "fun_name")(*args, **kwargs)


from . import functools
from . import it
from . import jaxlib_utils

logger: Any = None
from . import logging


def maybe_named_axis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for maybe_named_axis."""
    return getattr(_ops, "maybe_named_axis")(*args, **kwargs)


def memoize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for memoize."""
    return getattr(_ops, "memoize")(*args, **kwargs)


def merge_lists(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for merge_lists."""
    return getattr(_ops, "merge_lists")(*args, **kwargs)


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for moveaxis."""
    return getattr(_ops, "moveaxis")(*args, **kwargs)


from . import np
from . import operator


def overload(*args: Any, **kwargs: Any) -> Any:
    """Decorator for overloaded functions/methods."""
    return getattr(_ops, "overload")(*args, **kwargs)


class partial:
    """partial(func, *args, **keywords) - new function with partial application"""

    pass


def partition_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partition_list."""
    return getattr(_ops, "partition_list")(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_map."""
    return getattr(_ops, "safe_map")(*args, **kwargs)


def safe_zip(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_zip."""
    return getattr(_ops, "safe_zip")(*args, **kwargs)


def set_module(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for set_module."""
    return getattr(_ops, "set_module")(*args, **kwargs)


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


def stable_unique(*args: Any, **kwargs: Any) -> Any:
    """Returns unique elements from `it` in the order of occurrence."""
    return getattr(_ops, "stable_unique")(*args, **kwargs)


def subs_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subs_list."""
    return getattr(_ops, "subs_list")(*args, **kwargs)


def subs_list2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subs_list2."""
    return getattr(_ops, "subs_list2")(*args, **kwargs)


def subvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subvals."""
    return getattr(_ops, "subvals")(*args, **kwargs)


def toposort(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for toposort."""
    return getattr(_ops, "toposort")(*args, **kwargs)


def tuple_delete(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_delete."""
    return getattr(_ops, "tuple_delete")(*args, **kwargs)


def tuple_insert(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_insert."""
    return getattr(_ops, "tuple_insert")(*args, **kwargs)


def tuple_update(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_update."""
    return getattr(_ops, "tuple_update")(*args, **kwargs)


def unflatten(*args: Any, **kwargs: Any) -> Any:
    """Splits `xs` into subsequences of lengths `ns`."""
    return getattr(_ops, "unflatten")(*args, **kwargs)


def unzip2(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-2 tuples into two tuples."""
    return getattr(_ops, "unzip2")(*args, **kwargs)


def unzip3(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-3 tuples into three tuples."""
    return getattr(_ops, "unzip3")(*args, **kwargs)


def use_cpp_class(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to replace a python class with its C++ version"""
    return getattr(_ops, "use_cpp_class")(*args, **kwargs)


def use_cpp_method(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to exclude methods from the set that are forwarded to C++ class"""
    return getattr(_ops, "use_cpp_method")(*args, **kwargs)


from . import weakref


def weakref_lru_cache(*args: Any, **kwargs: Any) -> Any:
    """Least recently used cache decorator with weakref support."""
    return getattr(_ops, "weakref_lru_cache")(*args, **kwargs)


def wrap_name(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for wrap_name."""
    return getattr(_ops, "wrap_name")(*args, **kwargs)


def wraps(*args: Any, **kwargs: Any) -> Any:
    """Like functools.wraps, but with finer-grained control over the name and docstring"""
    return getattr(_ops, "wraps")(*args, **kwargs)


from . import xc
