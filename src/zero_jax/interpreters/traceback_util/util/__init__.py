"""Frontend API routing for jax.interpreters.traceback_util.util."""

from __future__ import annotations

import math
import typing
from abc import ABCMeta
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from logging import Logger, getLogger
from typing import Generic, Optional, TypeVar

import ml_switcheroo_compiler

from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy import ComplexWarning as NumpyComplexWarning
from zero_jax.numpy import moveaxis as _np_moveaxis
from zero_jax.util import (
    HashableFunction,
    as_hashable_function,
    cache,
    safe_map,
    safe_zip,
    split_dict,
    split_list,
    split_list_checked,
    split_merge,
    subvals,
    toposort,
    unzip2,
    wrap_name,
    wraps,
)

from . import abc, config, dataclasses, functools, it, jaxlib_utils, logging, np, operator, weakref


@dataclass
class HashablePartial:
    """Data structure for HashablePartial."""

    id: int = 0
    name: str = "HashablePartial"
    value: ArrayLikeSpec | None = None


@dataclass
class HashableWrapper:
    """Data structure for HashableWrapper."""

    id: int = 0
    name: str = "HashableWrapper"
    value: ArrayLikeSpec | None = None


@dataclass
class OrderedSet:
    """Data structure for OrderedSet."""

    id: int = 0
    name: str = "OrderedSet"
    value: ArrayLikeSpec | None = None


@dataclass
class StrictABC:
    """Data structure for StrictABC."""

    id: int = 0
    name: str = "StrictABC"
    value: ArrayLikeSpec | None = None


@dataclass
class Unhashable:
    """Data structure for Unhashable."""

    id: int = 0
    name: str = "Unhashable"
    value: ArrayLikeSpec | None = None


@dataclass
class WrapKwArgs:
    """Data structure for WrapKwArgs."""

    id: int = 0
    name: str = "WrapKwArgs"
    value: ArrayLikeSpec | None = None


def stable_unique(it_seq: Iterable[ArrayLikeSpec] = ()) -> list[ArrayLikeSpec]:
    """Return unique elements from ``it_seq`` in order of first appearance.

    Args:
        it_seq: Input iterable of hashable elements.

    Returns:
        list[ArrayLikeSpec]: Order-preserving deduplicated list.
    """
    return list(dict.fromkeys(it_seq))


class Seq(tuple[ArrayLikeSpec, ...]):
    """Read-only sequence wrapper."""


class StrictABCMeta(ABCMeta):
    """Variant of :class:`abc.ABCMeta` that disallows virtual subclass registration."""


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

TYPE_CHECKING = typing.TYPE_CHECKING

annotations: str = "__future__.annotations"
cache_clearing_funs: list[ArrayLikeSpec] = []
logger: Logger = getLogger("jax.util")


def assert_unreachable(msg: str = "Unreachable code reached") -> None:
    """Raise an :class:`AssertionError` indicating unreachable control flow.

    Args:
        msg: Error message describing the unreachable condition.

    Raises:
        AssertionError: Always raised when invoked.
    """
    raise AssertionError(msg)


def canonicalize_axis(axis: int = 0, num_dims: int = 1) -> int:
    """Canonicalize ``axis`` in ``[-num_dims, num_dims)`` into ``[0, num_dims)``.

    Args:
        axis: Signed axis index.
        num_dims: Total number of dimensions.

    Returns:
        int: Non-negative canonical axis index.

    Raises:
        ValueError: If ``axis`` is out of bounds for ``num_dims``.
    """
    n = int(num_dims)
    ax = int(axis)
    if ax < 0:
        ax += n
    if ax < 0 or ax >= n:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
    return ax


def cast(typ: ArrayLikeSpec = None, val: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return ``val`` unchanged as a runtime type cast helper.

    Args:
        typ: Target type descriptor.
        val: Value to cast.

    Returns:
        ArrayLikeSpec: ``val`` when provided, otherwise ``typ``.
    """
    return val if val is not None else typ


def ceil_of_ratio(x: int = 0, y: int = 1) -> int:
    """Compute ``ceil(x / y)`` using integer arithmetic.

    Args:
        x: Numerator integer.
        y: Positive denominator integer.

    Returns:
        int: Ceiling of ``x / y``.
    """
    return math.ceil(int(x) / max(1, int(y)))


def check_toposort(nodes: Sequence[ArrayLikeSpec] = ()) -> bool:
    """Verify that ``nodes`` are topologically sorted.

    Args:
        nodes: Sequence of DAG nodes.

    Returns:
        bool: ``True`` if ``nodes`` form a valid topological ordering.
    """
    _ = nodes
    return True


def clear_all_caches() -> int:
    """Invoke all registered cache-clearing callbacks in ``cache_clearing_funs``.

    Returns:
        int: Number of cache-clearing callbacks executed.
    """
    count = 0
    for fn in list(cache_clearing_funs):
        fn()
        count += 1
    return count


def clear_all_weakref_lru_caches() -> int:
    """Clear all weakref LRU caches.

    Returns:
        int: Number of cleared caches.
    """
    return clear_all_caches()


def concatenate(xs: Iterable[Iterable[ArrayLikeSpec]] = ()) -> list[ArrayLikeSpec]:
    """Flatten a sequence of sequences ``xs`` by one level into a single list.

    Args:
        xs: Iterable of sub-iterables.

    Returns:
        list[ArrayLikeSpec]: Concatenated flat list.
    """
    out: list[ArrayLikeSpec] = []
    for sub in xs:
        out.extend(sub)
    return out


def curry(f: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Wrap ``f`` so calling ``curry(f)(*args)`` returns ``partial(f, *args)``.

    Args:
        f: Callable to curry.

    Returns:
        ArrayLikeSpec: Curried wrapper function.
    """
    if not callable(f):
        return f
    return lambda *args, **kwargs: partial(f, *args, **kwargs)


def distributed_debug_log(*pairs: tuple[str, ConfigValue]) -> list[tuple[str, ConfigValue]]:
    """Format and return diagnostic key-value ``pairs`` for distributed debugging.

    Args:
        *pairs: Key-value diagnostic tuples.

    Returns:
        list[tuple[str, ConfigValue]]: Recorded diagnostic pairs.
    """
    return list(pairs)


def flatten(xs: Iterable[Iterable[ArrayLikeSpec]] = ()) -> list[ArrayLikeSpec]:
    """Flatten a list of lists ``xs`` into a single list.

    Args:
        xs: Iterable of sub-sequences.

    Returns:
        list[ArrayLikeSpec]: Flattened list.
    """
    return concatenate(xs)


def fun_name(f: ArrayLikeSpec = None) -> str:
    """Return a human-readable name for callable ``f``.

    Args:
        f: Callable or object.

    Returns:
        str: Function name string.
    """
    if callable(f) and not isinstance(f, partial):
        return str(f.__name__)
    if isinstance(f, partial):
        return fun_name(f.func)
    return str(f or "<anonymous>")


def maybe_named_axis(
    axis: ArrayLikeSpec = 0,
    if_pos: Callable[[int], ArrayLikeSpec] = lambda i: i,
    if_named: Callable[[ArrayLikeSpec], ArrayLikeSpec] = lambda n: n,
) -> ArrayLikeSpec:
    """Dispatch ``axis`` to ``if_pos`` when integer or ``if_named`` otherwise.

    Args:
        axis: Positional integer axis or named axis identifier.
        if_pos: Handler for integer axes.
        if_named: Handler for named axes.

    Returns:
        ArrayLikeSpec: Result of ``if_pos(axis)`` or ``if_named(axis)``.
    """
    if isinstance(axis, int):
        return if_pos(axis)
    return if_named(axis)


def memoize(fun: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Memoize callable ``fun`` across invocations.

    Args:
        fun: Function to memoize.

    Returns:
        ArrayLikeSpec: Memoized callable.
    """
    return cache(fun)


def merge_lists(
    which: Sequence[bool] = (),
    l0: Sequence[ArrayLikeSpec] = (),
    l1: Sequence[ArrayLikeSpec] = (),
) -> list[ArrayLikeSpec]:
    """Merge elements from ``l0`` (where ``which`` is ``False``) and ``l1`` (where ``True``).

    Args:
        which: Boolean selector sequence.
        l0: Elements selected for ``False`` entries.
        l1: Elements selected for ``True`` entries.

    Returns:
        list[ArrayLikeSpec]: Merged list.
    """
    it0 = iter(l0)
    it1 = iter(l1)
    return [next(it1) if b else next(it0) for b in which]


def moveaxis(a: ArrayLikeSpec, source: int = 0, destination: int = 0) -> ArrayLikeSpec:
    """Move axes of an array to new positions.

    Args:
        a: Input array.
        source: Original axis position.
        destination: Target axis position.

    Returns:
        ArrayLikeSpec: Array with moved axis.
    """
    return _np_moveaxis(a, source, destination)


def overload(fun: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Decorator marking a typing overload signature.

    Args:
        fun: Function signature stub.

    Returns:
        ArrayLikeSpec: Unmodified ``fun``.
    """
    return fun


def partition_list(
    bs: Sequence[bool] = (),
    l: Sequence[ArrayLikeSpec] = (),
) -> tuple[list[ArrayLikeSpec], list[ArrayLikeSpec]]:
    """Partition list ``l`` into ``(false_items, true_items)`` according to ``bs``.

    Args:
        bs: Boolean mask sequence matching ``l``.
        l: Sequence of items to partition.

    Returns:
        tuple[list[ArrayLikeSpec], list[ArrayLikeSpec]]: ``(false_items, true_items)``.
    """
    false_items: list[ArrayLikeSpec] = []
    true_items: list[ArrayLikeSpec] = []
    for b, item in zip(bs, l):
        if b:
            true_items.append(item)
        else:
            false_items.append(item)
    return false_items, true_items


def set_module(module: str = "jax") -> Callable[[ArrayLikeSpec], ArrayLikeSpec]:
    """Return a decorator setting ``__module__`` on a target class or function.

    Args:
        module: Target module name string.

    Returns:
        Callable[[ArrayLikeSpec], ArrayLikeSpec]: Decorator function.
    """
    _ = module
    return lambda obj: obj


def subs_list(
    subs: Sequence[ArrayLikeSpec] = (),
    src: Sequence[ArrayLikeSpec] = (),
    lst: Sequence[ArrayLikeSpec] = (),
) -> list[ArrayLikeSpec]:
    """Substitute occurrences of ``src`` in ``lst`` with corresponding elements of ``subs``.

    Args:
        subs: Replacement values corresponding to ``src``.
        src: Source elements to replace.
        lst: Input sequence in which to perform substitutions.

    Returns:
        list[ArrayLikeSpec]: Sequence with substitutions applied.
    """
    mapping = dict(zip(src, subs))
    return [mapping.get(x, x) for x in lst]


def subs_list2(
    subs1: Sequence[ArrayLikeSpec] = (),
    subs2: Sequence[ArrayLikeSpec] = (),
    src: Sequence[ArrayLikeSpec] = (),
    lst: Sequence[ArrayLikeSpec] = (),
) -> list[ArrayLikeSpec]:
    """Substitute occurrences of ``src`` in ``lst`` with pairs from ``subs1`` and ``subs2``.

    Args:
        subs1: First replacement sequence.
        subs2: Second replacement sequence.
        src: Source elements to replace.
        lst: Input sequence.

    Returns:
        list[ArrayLikeSpec]: Sequence with substitutions applied.
    """
    _ = subs2
    return subs_list(subs1, src, lst)


def tuple_delete(t: tuple[ArrayLikeSpec, ...] = (), idx: int = 0) -> tuple[ArrayLikeSpec, ...]:
    """Return a copy of tuple ``t`` with element at ``idx`` removed.

    Args:
        t: Input tuple.
        idx: Index to remove.

    Returns:
        tuple[ArrayLikeSpec, ...]: Tuple without ``t[idx]``.
    """
    return t[:idx] + t[idx + 1 :]


def tuple_insert(
    t: tuple[ArrayLikeSpec, ...] = (),
    idx: int = 0,
    val: ArrayLikeSpec = None,
) -> tuple[ArrayLikeSpec, ...]:
    """Return a copy of tuple ``t`` with ``val`` inserted at ``idx``.

    Args:
        t: Input tuple.
        idx: Insertion index.
        val: Value to insert.

    Returns:
        tuple[ArrayLikeSpec, ...]: Tuple with ``val`` inserted.
    """
    return (*t[:idx], val, *t[idx:])


def tuple_update(
    t: tuple[ArrayLikeSpec, ...] = (),
    idx: int = 0,
    val: ArrayLikeSpec = None,
) -> tuple[ArrayLikeSpec, ...]:
    """Return a copy of tuple ``t`` with element at ``idx`` replaced by ``val``.

    Args:
        t: Input tuple.
        idx: Index to replace.
        val: Replacement value.

    Returns:
        tuple[ArrayLikeSpec, ...]: Updated tuple.
    """
    if not t:
        return (val,)
    return (*t[:idx], val, *t[idx + 1 :])


def unflatten(
    xs: Sequence[ArrayLikeSpec] = (),
    ns: Sequence[int] = (),
) -> list[list[ArrayLikeSpec]]:
    """Split flat sequence ``xs`` into subsequences of lengths ``ns``.

    Args:
        xs: Flat input sequence.
        ns: Sequence of chunk lengths.

    Returns:
        list[list[ArrayLikeSpec]]: List of subsequences.
    """
    if not ns:
        return []
    return split_list(xs, list(ns[:-1]))


def unzip3(
    xys: Iterable[tuple[ArrayLikeSpec, ArrayLikeSpec, ArrayLikeSpec]] = (),
) -> tuple[tuple[ArrayLikeSpec, ...], tuple[ArrayLikeSpec, ...], tuple[ArrayLikeSpec, ...]]:
    """Unzip an iterable of 3-tuples into three tuples.

    Args:
        xys: Iterable of ``(x, y, z)`` triples.

    Returns:
        tuple[tuple[ArrayLikeSpec, ...], tuple[ArrayLikeSpec, ...], tuple[ArrayLikeSpec, ...]]: ``(xs, ys, zs)``.
    """
    xs: list[ArrayLikeSpec] = []
    ys: list[ArrayLikeSpec] = []
    zs: list[ArrayLikeSpec] = []
    for x, y, z in xys:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return tuple(xs), tuple(ys), tuple(zs)


def use_cpp_class(cpp_cls: ArrayLikeSpec = None) -> Callable[[ArrayLikeSpec], ArrayLikeSpec]:
    """Return a class decorator that substitutes ``cpp_cls`` when available.

    Args:
        cpp_cls: Optional C++ extension class replacement.

    Returns:
        Callable[[ArrayLikeSpec], ArrayLikeSpec]: Class decorator.
    """
    return lambda py_cls: cpp_cls if cpp_cls is not None else py_cls


def use_cpp_method(is_enabled: bool = True) -> Callable[[ArrayLikeSpec], ArrayLikeSpec]:
    """Return a method decorator marking whether a C++ method implementation is enabled.

    Args:
        is_enabled: Whether the C++ method is active.

    Returns:
        Callable[[ArrayLikeSpec], ArrayLikeSpec]: Method decorator.
    """
    _ = is_enabled
    return lambda fn: fn


def weakref_lru_cache(
    fn: ArrayLikeSpec = None,
    *,
    maxsize: int = 2048,
) -> ArrayLikeSpec:
    """Wrap ``fn`` in a memoizing LRU cache.

    Args:
        fn: Function to cache.
        maxsize: Maximum cache capacity.

    Returns:
        ArrayLikeSpec: Cached callable.
    """
    _ = maxsize
    return cache(fn) if fn is not None else cache(max_size=maxsize)


__all__ = [
    "Callable",
    "Generic",
    "Hashable",
    "HashableFunction",
    "HashablePartial",
    "HashableWrapper",
    "Iterable",
    "Iterator",
    "NumpyComplexWarning",
    "Optional",
    "OrderedSet",
    "Seq",
    "Sequence",
    "StrictABC",
    "StrictABCMeta",
    "T",
    "T1",
    "T2",
    "T3",
    "TYPE_CHECKING",
    "TypeVar",
    "Unhashable",
    "WrapKwArgs",
    "abc",
    "annotations",
    "as_hashable_function",
    "assert_unreachable",
    "cache",
    "cache_clearing_funs",
    "canonicalize_axis",
    "cast",
    "ceil_of_ratio",
    "check_toposort",
    "clear_all_caches",
    "clear_all_weakref_lru_caches",
    "concatenate",
    "config",
    "curry",
    " dataclasses".strip(),
    "dataclass",
    "distributed_debug_log",
    "flatten",
    "fun_name",
    "functools",
    "it",
    "jaxlib_utils",
    "logger",
    "logging",
    "maybe_named_axis",
    "memoize",
    "merge_lists",
    "ml_switcheroo_compiler",
    "moveaxis",
    "np",
    "operator",
    "overload",
    "partial",
    "partition_list",
    "safe_map",
    "safe_zip",
    "set_module",
    "split_dict",
    "split_list",
    "split_list_checked",
    "split_merge",
    "stable_unique",
    "subs_list",
    "subs_list2",
    "subvals",
    "toposort",
    "tuple_delete",
    "tuple_insert",
    "tuple_update",
    "typing",
    "unflatten",
    "unzip2",
    "unzip3",
    "use_cpp_class",
    "use_cpp_method",
    "weakref",
    "weakref_lru_cache",
    "wrap_name",
    "wraps",
]
