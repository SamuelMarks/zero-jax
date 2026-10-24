"""Frontend API routing for jax.interpreters.traceback_util.util."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class Generic:
    """Abstract base class for generic types."""

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


@dataclass
class Hashable:
    """Data structure for Hashable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Hashable"
    value: ArrayLikeSpec | None = None


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


@dataclass
class HashablePartial:
    """Data structure for HashablePartial.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "HashablePartial"
    value: ArrayLikeSpec | None = None


@dataclass
class HashableWrapper:
    """Data structure for HashableWrapper.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "HashableWrapper"
    value: ArrayLikeSpec | None = None


@dataclass
class Iterable:
    """Data structure for Iterable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Iterable"
    value: ArrayLikeSpec | None = None


@dataclass
class Iterator:
    """Data structure for Iterator.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Iterator"
    value: ArrayLikeSpec | None = None


class NumpyComplexWarning:
    """The warning raised when casting a complex dtype to a real dtype."""

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


@dataclass
class OrderedSet:
    """Data structure for OrderedSet.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "OrderedSet"
    value: ArrayLikeSpec | None = None


class Seq:
    """All the operations on a read-only sequence."""

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


@dataclass
class StrictABC:
    """Data structure for StrictABC.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "StrictABC"
    value: ArrayLikeSpec | None = None


class StrictABCMeta:
    """A variant of `abc.ABCMeta` which does not allow virtual subclasses."""

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


T = None

import typing

T1 = typing.TypeVar("T1")
T2 = typing.TypeVar("T2")
T3 = typing.TypeVar("T3")

TYPE_CHECKING = typing.TYPE_CHECKING


class TypeVar:
    """Type variable."""

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


@dataclass
class Unhashable:
    """Data structure for Unhashable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Unhashable"
    value: ArrayLikeSpec | None = None


@dataclass
class WrapKwArgs:
    """Data structure for WrapKwArgs.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "WrapKwArgs"
    value: ArrayLikeSpec | None = None


from . import abc

annotations: ArrayLikeSpec = None


def as_hashable_function(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for as_hashable_function.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.as_hashable_function(*args, **kwargs)


def assert_unreachable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for assert_unreachable.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.assert_unreachable(*args, **kwargs)


def cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for cache.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.cache(*args, **kwargs)


cache_clearing_funs: list[ArrayLikeSpec] = []


def canonicalize_axis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.canonicalize_axis(*args, **kwargs)


def cast(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cast a value to a type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cast(*args, **kwargs)


def ceil_of_ratio(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for ceil_of_ratio.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.ceil_of_ratio(*args, **kwargs)


def check_toposort(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for check_toposort.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.check_toposort(*args, **kwargs)


def clear_all_caches(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for clear_all_caches.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.clear_all_caches(*args, **kwargs)


def clear_all_weakref_lru_caches(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for clear_all_weakref_lru_caches.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.clear_all_weakref_lru_caches(*args, **kwargs)


def concatenate(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Concatenates/flattens a list of lists.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.concatenate(*args, **kwargs)


from . import config


def curry(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Curries arguments of f, returning a function on any remaining arguments.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.curry(*args, **kwargs)


from . import dataclasses


def distributed_debug_log(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Format and log `pairs` if config.jax_distributed_debug is enabled.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.distributed_debug_log(*args, **kwargs)


def flatten(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Concatenates/flattens a list of lists.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.flatten(*args, **kwargs)


def fun_name(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for fun_name.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.fun_name(*args, **kwargs)


from . import functools, it, jaxlib_utils

logger: ArrayLikeSpec = None
from . import logging


def maybe_named_axis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for maybe_named_axis.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.maybe_named_axis(*args, **kwargs)


def memoize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for memoize.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.memoize(*args, **kwargs)


def merge_lists(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for merge_lists.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.merge_lists(*args, **kwargs)


def moveaxis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for moveaxis.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.moveaxis(*args, **kwargs)


from . import np, operator


def overload(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Decorator for overloaded functions/methods.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.overload(*args, **kwargs)


class partial:
    """partial(func, *args, **keywords) - new function with partial application."""

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


def partition_list(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for partition_list.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.partition_list(*args, **kwargs)


def safe_map(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_map.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.safe_map(*args, **kwargs)


def safe_zip(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_zip.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.safe_zip(*args, **kwargs)


def set_module(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for set_module.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.set_module(*args, **kwargs)


def split_dict(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_dict.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.split_dict(*args, **kwargs)


def split_list(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.split_list(*args, **kwargs)


def split_list_checked(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list_checked.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.split_list_checked(*args, **kwargs)


def split_merge(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_merge.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.split_merge(*args, **kwargs)


def stable_unique(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns unique elements from `it` in the order of occurrence.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.stable_unique(*args, **kwargs)


def subs_list(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for subs_list.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.subs_list(*args, **kwargs)


def subs_list2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for subs_list2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.subs_list2(*args, **kwargs)


def subvals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for subvals.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.subvals(*args, **kwargs)


def toposort(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for toposort.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.toposort(*args, **kwargs)


def tuple_delete(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for tuple_delete.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.tuple_delete(*args, **kwargs)


def tuple_insert(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for tuple_insert.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.tuple_insert(*args, **kwargs)


def tuple_update(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for tuple_update.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.tuple_update(*args, **kwargs)


def unflatten(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Splits `xs` into subsequences of lengths `ns`.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.unflatten(*args, **kwargs)


def unzip2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Unzip sequence of length-2 tuples into two tuples.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.unzip2(*args, **kwargs)


def unzip3(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Unzip sequence of length-3 tuples into three tuples.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.unzip3(*args, **kwargs)


def use_cpp_class(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """A helper decorator to replace a python class with its C++ version.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.use_cpp_class(*args, **kwargs)


def use_cpp_method(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """A helper decorator to exclude methods from the set that are forwarded to C++ class.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.use_cpp_method(*args, **kwargs)


from . import weakref


def weakref_lru_cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Least recently used cache decorator with weakref support.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.weakref_lru_cache(*args, **kwargs)


def wrap_name(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for wrap_name.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
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


import typing

import ml_switcheroo_compiler

Sequence = typing.Sequence


class Callable:
    """Stub class for Callable."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Callable.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs
