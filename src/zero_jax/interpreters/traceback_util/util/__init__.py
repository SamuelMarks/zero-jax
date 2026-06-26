"""Frontend API routing for jax.interpreters.traceback_util.util."""

from typing import Any


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
    raise NotImplementedError("as_hashable_function not yet implemented in zero-jax")


def assert_unreachable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for assert_unreachable."""
    raise NotImplementedError("assert_unreachable not yet implemented in zero-jax")


def cache(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for cache."""
    raise NotImplementedError("cache not yet implemented in zero-jax")


cache_clearing_funs: Any = None


def canonicalize_axis(*args: Any, **kwargs: Any) -> Any:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    raise NotImplementedError("canonicalize_axis not yet implemented in zero-jax")


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    raise NotImplementedError("cast not yet implemented in zero-jax")


def ceil_of_ratio(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ceil_of_ratio."""
    raise NotImplementedError("ceil_of_ratio not yet implemented in zero-jax")


def check_toposort(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_toposort."""
    raise NotImplementedError("check_toposort not yet implemented in zero-jax")


def clear_all_caches(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for clear_all_caches."""
    raise NotImplementedError("clear_all_caches not yet implemented in zero-jax")


def clear_all_weakref_lru_caches(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for clear_all_weakref_lru_caches."""
    raise NotImplementedError(
        "clear_all_weakref_lru_caches not yet implemented in zero-jax"
    )


def concatenate(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    raise NotImplementedError("concatenate not yet implemented in zero-jax")


from . import config


def curry(*args: Any, **kwargs: Any) -> Any:
    """Curries arguments of f, returning a function on any remaining arguments."""
    raise NotImplementedError("curry not yet implemented in zero-jax")


from . import dataclasses


def distributed_debug_log(*args: Any, **kwargs: Any) -> Any:
    """Format and log `pairs` if config.jax_distributed_debug is enabled."""
    raise NotImplementedError("distributed_debug_log not yet implemented in zero-jax")


def flatten(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    raise NotImplementedError("flatten not yet implemented in zero-jax")


def fun_name(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for fun_name."""
    raise NotImplementedError("fun_name not yet implemented in zero-jax")


from . import functools
from . import it
from . import jaxlib_utils

logger: Any = None
from . import logging


def maybe_named_axis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for maybe_named_axis."""
    raise NotImplementedError("maybe_named_axis not yet implemented in zero-jax")


def memoize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for memoize."""
    raise NotImplementedError("memoize not yet implemented in zero-jax")


def merge_lists(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for merge_lists."""
    raise NotImplementedError("merge_lists not yet implemented in zero-jax")


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for moveaxis."""
    raise NotImplementedError("moveaxis not yet implemented in zero-jax")


from . import np
from . import operator


def overload(*args: Any, **kwargs: Any) -> Any:
    """Decorator for overloaded functions/methods."""
    raise NotImplementedError("overload not yet implemented in zero-jax")


class partial:
    """partial(func, *args, **keywords) - new function with partial application"""

    pass


def partition_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partition_list."""
    raise NotImplementedError("partition_list not yet implemented in zero-jax")


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_map."""
    raise NotImplementedError("safe_map not yet implemented in zero-jax")


def safe_zip(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for safe_zip."""
    raise NotImplementedError("safe_zip not yet implemented in zero-jax")


def set_module(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for set_module."""
    raise NotImplementedError("set_module not yet implemented in zero-jax")


def split_dict(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_dict."""
    raise NotImplementedError("split_dict not yet implemented in zero-jax")


def split_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_list."""
    raise NotImplementedError("split_list not yet implemented in zero-jax")


def split_list_checked(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_list_checked."""
    raise NotImplementedError("split_list_checked not yet implemented in zero-jax")


def split_merge(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for split_merge."""
    raise NotImplementedError("split_merge not yet implemented in zero-jax")


def stable_unique(*args: Any, **kwargs: Any) -> Any:
    """Returns unique elements from `it` in the order of occurrence."""
    raise NotImplementedError("stable_unique not yet implemented in zero-jax")


def subs_list(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subs_list."""
    raise NotImplementedError("subs_list not yet implemented in zero-jax")


def subs_list2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subs_list2."""
    raise NotImplementedError("subs_list2 not yet implemented in zero-jax")


def subvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subvals."""
    raise NotImplementedError("subvals not yet implemented in zero-jax")


def toposort(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for toposort."""
    raise NotImplementedError("toposort not yet implemented in zero-jax")


def tuple_delete(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_delete."""
    raise NotImplementedError("tuple_delete not yet implemented in zero-jax")


def tuple_insert(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_insert."""
    raise NotImplementedError("tuple_insert not yet implemented in zero-jax")


def tuple_update(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tuple_update."""
    raise NotImplementedError("tuple_update not yet implemented in zero-jax")


def unflatten(*args: Any, **kwargs: Any) -> Any:
    """Splits `xs` into subsequences of lengths `ns`."""
    raise NotImplementedError("unflatten not yet implemented in zero-jax")


def unzip2(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-2 tuples into two tuples."""
    raise NotImplementedError("unzip2 not yet implemented in zero-jax")


def unzip3(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-3 tuples into three tuples."""
    raise NotImplementedError("unzip3 not yet implemented in zero-jax")


def use_cpp_class(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to replace a python class with its C++ version"""
    raise NotImplementedError("use_cpp_class not yet implemented in zero-jax")


def use_cpp_method(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to exclude methods from the set that are forwarded to C++ class"""
    raise NotImplementedError("use_cpp_method not yet implemented in zero-jax")


from . import weakref


def weakref_lru_cache(*args: Any, **kwargs: Any) -> Any:
    """Least recently used cache decorator with weakref support."""
    raise NotImplementedError("weakref_lru_cache not yet implemented in zero-jax")


def wrap_name(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for wrap_name."""
    raise NotImplementedError("wrap_name not yet implemented in zero-jax")


def wraps(*args: Any, **kwargs: Any) -> Any:
    """Like functools.wraps, but with finer-grained control over the name and docstring"""
    raise NotImplementedError("wraps not yet implemented in zero-jax")


from . import xc
