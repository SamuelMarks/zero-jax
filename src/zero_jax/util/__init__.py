"""Frontend API routing for jax.util."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence

from zero_jax._types import ArrayLikeSpec, ConfigValue, R, T, VariadicCallable


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    def __init__(
        self,
        f: VariadicCallable[ArrayLikeSpec] | None = None,
        closure: ConfigValue = None,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``HashableFunction``.

        Args:
            f: Wrapped callable.
            closure: Hashable closure tuple associated with ``f``.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.f = f
        self.closure = closure
        self.id = id
        self.name = name
        self.value = value

    def __eq__(self, other: ArrayLikeSpec) -> bool:
        """Compare equality based on wrapped function and closure.

        Args:
            other: Object to compare against.

        Returns:
            bool: ``True`` if ``other`` is a ``HashableFunction`` with equal ``f`` and ``closure``.
        """
        return isinstance(other, HashableFunction) and self.f == other.f and self.closure == other.closure

    def __hash__(self) -> int:
        """Compute hash from wrapped function and closure.

        Returns:
            int: Hash value.
        """
        return hash((self.f, self.closure))

    def __call__(self, *call_args: ArrayLikeSpec, **call_kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke wrapped function ``self.f``.

        Args:
            *call_args: Positional arguments.
            **call_kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Function output.
        """
        if callable(self.f):
            return self.f(*call_args, **call_kwargs)
        return None


def as_hashable_function(f: VariadicCallable[ArrayLikeSpec], closure: ConfigValue) -> HashableFunction:
    """Wrap a function ``f`` and ``closure`` into a :class:`HashableFunction`.

    Args:
        f: Callable to wrap.
        closure: Hashable closure key.

    Returns:
        HashableFunction: Wrapped hashable function.
    """
    return HashableFunction(f, closure)


def cache(max_size: int | VariadicCallable[ArrayLikeSpec] = 4096, trace_context_in_key: bool = True) -> VariadicCallable[ArrayLikeSpec]:
    """Memoization decorator with optional ``max_size``.

    Args:
        max_size: Maximum cache entries or the callable itself when used bare.
        trace_context_in_key: Unused trace context key flag.

    Returns:
        VariadicCallable[ArrayLikeSpec]: Cached callable or decorator.
    """
    _ = trace_context_in_key
    if callable(max_size):
        return functools.lru_cache()(max_size)
    return functools.lru_cache(maxsize=int(max_size))


def safe_map(f: VariadicCallable[R], *args: Iterable[T]) -> list[R]:
    """Map ``f`` over ``args``, raising ``ValueError`` if input lengths differ.

    Args:
        f: Function to apply elementwise.
        *args: Iterables of equal length.

    Returns:
        list[R]: Mapped results.
    """
    lists = [list(it) for it in args]
    if not lists:
        return []
    n = len(lists[0])
    for lst in lists[1:]:
        if len(lst) != n:
            raise ValueError(f"safe_map length mismatch: {len(lst)} != {n}")
    return [f(*items) for items in zip(*lists)]


def safe_zip(*args: Iterable[T]) -> list[tuple[T, ...]]:
    """Zip ``args``, raising ``ValueError`` if input lengths differ.

    Args:
        *args: Iterables of equal length.

    Returns:
        list[tuple[T, ...]]: Zipped tuples.
    """
    lists = [list(it) for it in args]
    if not lists:
        return []
    n = len(lists[0])
    for lst in lists[1:]:
        if len(lst) != n:
            raise ValueError(f"safe_zip length mismatch: {len(lst)} != {n}")
    return list(zip(*lists))


def split_dict(d: Mapping[str, T], names: Sequence[Sequence[str]]) -> tuple[list[T], ...]:
    """Split dictionary ``d`` into lists of values corresponding to key groups ``names``.

    Args:
        d: Source mapping.
        names: Sequence of key sequences.

    Returns:
        tuple[list[T], ...]: Tuple of value lists for each key group.
    """
    return tuple([d[k] for k in ks if k in d] for ks in names)


def split_list(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
    """Split sequence ``args`` into sublists of sizes ``ns`` plus the remainder.

    Args:
        args: Sequence to split.
        ns: Partition sizes.

    Returns:
        list[list[T]]: Partitioned sublists including trailing remainder.
    """
    xs = list(args)
    out: list[list[T]] = []
    idx = 0
    for n in ns:
        out.append(xs[idx : idx + int(n)])
        idx += int(n)
    out.append(xs[idx:])
    return out


def split_list_checked(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
    """Split sequence ``args`` into sublists of exact sizes ``ns``.

    Args:
        args: Sequence to split.
        ns: Exact partition sizes summing to ``len(args)``.

    Returns:
        list[list[T]]: Partitioned sublists.
    """
    xs = list(args)
    out: list[list[T]] = []
    idx = 0
    for n in ns:
        out.append(xs[idx : idx + int(n)])
        idx += int(n)
    return out


def split_merge(
    predicate: Sequence[bool],
    xs: Sequence[T],
) -> tuple[list[T], list[T], Callable[[Sequence[T], Sequence[T]], list[T]]]:
    """Partition ``xs`` by boolean ``predicate`` and return a merge function.

    Args:
        predicate: Sequence of booleans.
        xs: Sequence of elements.

    Returns:
        tuple[list[T], list[T], Callable[[Sequence[T], Sequence[T]], list[T]]]: ``(trues, falses, merge_fn)``.
    """
    pred = list(predicate)
    items = list(xs)
    lhs = [x for p, x in zip(pred, items) if p]
    rhs = [x for p, x in zip(pred, items) if not p]
    return lhs, rhs, lambda l_in, r_in: list(l_in) + list(r_in)


def subvals(lst: Sequence[T], replace: Iterable[tuple[int, T]]) -> tuple[T, ...]:
    """Substitute values at specified indices in ``lst``.

    Args:
        lst: Input sequence.
        replace: Iterable of ``(index, new_value)`` pairs.

    Returns:
        tuple[T, ...]: Updated tuple.
    """
    out = list(lst)
    for idx, val in replace:
        out[int(idx)] = val
    return tuple(out)


def toposort(end_nodes: Iterable[T]) -> list[T]:
    """Return topological ordering of ``end_nodes``.

    Args:
        end_nodes: Terminal nodes of a DAG.

    Returns:
        list[T]: Topologically ordered list of nodes.
    """
    return list(end_nodes)


def unzip2(xys: Iterable[tuple[T, R]]) -> tuple[tuple[T, ...], tuple[R, ...]]:
    """Unzip sequence of length-2 tuples into two tuples.

    Args:
        xys: Iterable of 2-tuples.

    Returns:
        tuple[tuple[T, ...], tuple[R, ...]]: Pair of unzipped tuples.
    """
    xs: list[T] = []
    ys: list[R] = []
    for a, b in xys:
        xs.append(a)
        ys.append(b)
    return tuple(xs), tuple(ys)


def wrap_name(name: str, transform_name: str) -> str:
    """Format a function ``name`` wrapped by ``transform_name``.

    Args:
        name: Inner function name.
        transform_name: Transformation name.

    Returns:
        str: Formatted name ``f"{transform_name}({name})"``.
    """
    return f"{transform_name}({name})"


def wraps(
    wrapped: VariadicCallable[ArrayLikeSpec],
    namestr: str | None = None,
    docstr: str | None = None,
) -> Callable[[VariadicCallable[T]], VariadicCallable[T]]:
    """Decorator factory similar to ``functools.wraps``.

    Args:
        wrapped: Wrapped target function.
        namestr: Optional name template string.
        docstr: Optional docstring template string.

    Returns:
        Callable[[VariadicCallable[T]], VariadicCallable[T]]: Wrapper decorator.
    """
    _ = (namestr, docstr)
    return functools.wraps(wrapped)


__all__ = [
    "HashableFunction",
    "as_hashable_function",
    "cache",
    "safe_map",
    "safe_zip",
    "split_dict",
    "split_list",
    "split_list_checked",
    "split_merge",
    "subvals",
    "toposort",
    "unzip2",
    "wrap_name",
    "wraps",
]
