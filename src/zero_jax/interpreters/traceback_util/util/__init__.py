"""Frontend API routing for jax.interpreters.traceback_util.util."""

from dataclasses import dataclass
from typing import Any, Optional

import zero_jax._compiler_proxy_ops as _ops


class Generic:
    """Abstract base class for generic types."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class Hashable:
    """Data structure for Hashable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Hashable"
    value: Optional[Any] = None


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class HashablePartial:
    """Data structure for HashablePartial.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "HashablePartial"
    value: Optional[Any] = None


@dataclass
class HashableWrapper:
    """Data structure for HashableWrapper.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "HashableWrapper"
    value: Optional[Any] = None


@dataclass
class Iterable:
    """Data structure for Iterable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Iterable"
    value: Optional[Any] = None


@dataclass
class Iterator:
    """Data structure for Iterator.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Iterator"
    value: Optional[Any] = None


class NumpyComplexWarning:
    """The warning raised when casting a complex dtype to a real dtype."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class OrderedSet:
    """Data structure for OrderedSet.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "OrderedSet"
    value: Optional[Any] = None


class Seq:
    """All the operations on a read-only sequence."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class StrictABC:
    """Data structure for StrictABC.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "StrictABC"
    value: Optional[Any] = None


class StrictABCMeta:
    """A variant of `abc.ABCMeta` which does not allow virtual subclasses."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


T = None

import typing

T1 = typing.TypeVar("T1")
T2 = typing.TypeVar("T2")
T3 = typing.TypeVar("T3")

TYPE_CHECKING = typing.TYPE_CHECKING


class TypeVar:
    """Type variable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class Unhashable:
    """Data structure for Unhashable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Unhashable"
    value: Optional[Any] = None


@dataclass
class WrapKwArgs:
    """Data structure for WrapKwArgs.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "WrapKwArgs"
    value: Optional[Any] = None


from . import abc

annotations: Any = None


def as_hashable_function(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for as_hashable_function.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.as_hashable_function(*args, **kwargs)


def assert_unreachable(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for assert_unreachable.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.assert_unreachable(*args, **kwargs)


def cache(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for cache.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.cache(*args, **kwargs)


cache_clearing_funs: typing.List[typing.Any] = []


def canonicalize_axis(*args: Any, **kwargs: Any) -> Any:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
    return _ops.canonicalize_axis(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return _ops.cast(*args, **kwargs)


def ceil_of_ratio(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for ceil_of_ratio.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.ceil_of_ratio(*args, **kwargs)


def check_toposort(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for check_toposort.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.check_toposort(*args, **kwargs)


def clear_all_caches(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for clear_all_caches.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.clear_all_caches(*args, **kwargs)


def clear_all_weakref_lru_caches(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for clear_all_weakref_lru_caches.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.clear_all_weakref_lru_caches(*args, **kwargs)


def concatenate(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    return _ops.concatenate(*args, **kwargs)


from . import config


def curry(*args: Any, **kwargs: Any) -> Any:
    """Curries arguments of f, returning a function on any remaining arguments."""
    return _ops.curry(*args, **kwargs)


from . import dataclasses


def distributed_debug_log(*args: Any, **kwargs: Any) -> Any:
    """Format and log `pairs` if config.jax_distributed_debug is enabled."""
    return _ops.distributed_debug_log(*args, **kwargs)


def flatten(*args: Any, **kwargs: Any) -> Any:
    """Concatenates/flattens a list of lists."""
    return _ops.flatten(*args, **kwargs)


def fun_name(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for fun_name.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.fun_name(*args, **kwargs)


from . import functools, it, jaxlib_utils

logger: Any = None
from . import logging


def maybe_named_axis(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for maybe_named_axis.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.maybe_named_axis(*args, **kwargs)


def memoize(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for memoize.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.memoize(*args, **kwargs)


def merge_lists(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for merge_lists.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.merge_lists(*args, **kwargs)


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for moveaxis.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.moveaxis(*args, **kwargs)


from . import np, operator


def overload(*args: Any, **kwargs: Any) -> Any:
    """Decorator for overloaded functions/methods."""
    return _ops.overload(*args, **kwargs)


class partial:
    """partial(func, *args, **keywords) - new function with partial application"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def partition_list(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for partition_list.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.partition_list(*args, **kwargs)


def safe_map(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for safe_map.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.safe_map(*args, **kwargs)


def safe_zip(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for safe_zip.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.safe_zip(*args, **kwargs)


def set_module(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for set_module.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.set_module(*args, **kwargs)


def split_dict(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_dict.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.split_dict(*args, **kwargs)


def split_list(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_list.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.split_list(*args, **kwargs)


def split_list_checked(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_list_checked.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.split_list_checked(*args, **kwargs)


def split_merge(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for split_merge.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.split_merge(*args, **kwargs)


def stable_unique(*args: Any, **kwargs: Any) -> Any:
    """Returns unique elements from `it` in the order of occurrence."""
    return _ops.stable_unique(*args, **kwargs)


def subs_list(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for subs_list.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.subs_list(*args, **kwargs)


def subs_list2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for subs_list2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.subs_list2(*args, **kwargs)


def subvals(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for subvals.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.subvals(*args, **kwargs)


def toposort(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for toposort.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.toposort(*args, **kwargs)


def tuple_delete(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for tuple_delete.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.tuple_delete(*args, **kwargs)


def tuple_insert(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for tuple_insert.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.tuple_insert(*args, **kwargs)


def tuple_update(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for tuple_update.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.tuple_update(*args, **kwargs)


def unflatten(*args: Any, **kwargs: Any) -> Any:
    """Splits `xs` into subsequences of lengths `ns`."""
    return _ops.unflatten(*args, **kwargs)


def unzip2(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-2 tuples into two tuples."""
    return _ops.unzip2(*args, **kwargs)


def unzip3(*args: Any, **kwargs: Any) -> Any:
    """Unzip sequence of length-3 tuples into three tuples."""
    return _ops.unzip3(*args, **kwargs)


def use_cpp_class(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to replace a python class with its C++ version"""
    return _ops.use_cpp_class(*args, **kwargs)


def use_cpp_method(*args: Any, **kwargs: Any) -> Any:
    """A helper decorator to exclude methods from the set that are forwarded to C++ class"""
    return _ops.use_cpp_method(*args, **kwargs)


from . import weakref


def weakref_lru_cache(*args: Any, **kwargs: Any) -> Any:
    """Least recently used cache decorator with weakref support."""
    return _ops.weakref_lru_cache(*args, **kwargs)


def wrap_name(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for wrap_name.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.wrap_name(*args, **kwargs)


def wraps(*args: Any, **kwargs: Any) -> Any:
    """Like functools.wraps, but with finer-grained control over the name and docstring"""
    return _ops.wraps(*args, **kwargs)


import typing

import ml_switcheroo_compiler

Sequence = typing.Sequence


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
