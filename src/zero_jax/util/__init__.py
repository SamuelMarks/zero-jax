"""Frontend API routing for jax.util."""

from __future__ import annotations

import functools

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class HashableFunction:
    """Decouples function equality and hash from its identity."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments ``(f, closure)``.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.f = args[0] if args else None
        self.closure = args[1] if len(args) > 1 else None
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs

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


def as_hashable_function(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for as_hashable_function.

    Args:
        *args: Positional arguments ``(f, closure)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if args:
        return HashableFunction(*args)
    return _ops.as_hashable_function(*args, **kwargs)


def cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for cache.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if args and callable(args[0]):
        return functools.lru_cache()(args[0])
    return _ops.cache(*args, **kwargs)


def safe_map(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_map.

    Args:
        *args: Positional arguments ``(f, *iterables)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2 and callable(args[0]):
        f = args[0]
        lists = [list(it) for it in args[1:]]
        n = len(lists[0])
        for lst in lists[1:]:
            if len(lst) != n:
                raise ValueError(f"safe_map length mismatch: {len(lst)} != {n}")
        return [f(*items) for items in zip(*lists)]
    return _ops.safe_map(*args, **kwargs)


def safe_zip(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for safe_zip.

    Args:
        *args: Positional arguments ``(*iterables)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if args:
        lists = [list(it) for it in args]
        n = len(lists[0])
        for lst in lists[1:]:
            if len(lst) != n:
                raise ValueError(f"safe_zip length mismatch: {len(lst)} != {n}")
        return list(zip(*lists))
    return _ops.safe_zip(*args, **kwargs)


def split_dict(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_dict.

    Args:
        *args: Positional arguments ``(d, names)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2 and isinstance(args[0], dict):
        d = args[0]
        return tuple([d[k] for k in ks if k in d] for ks in args[1])
    return _ops.split_dict(*args, **kwargs)


def split_list(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list.

    Args:
        *args: Positional arguments ``(args_list, ns)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2:
        xs = list(args[0])
        ns = list(args[1])
        out: list[list[ArrayLikeSpec]] = []
        idx = 0
        for n in ns:
            out.append(xs[idx : idx + int(n)])
            idx += int(n)
        out.append(xs[idx:])
        return out
    return _ops.split_list(*args, **kwargs)


def split_list_checked(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_list_checked.

    Args:
        *args: Positional arguments ``(args_list, ns)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2:
        xs = list(args[0])
        ns = list(args[1])
        out: list[list[ArrayLikeSpec]] = []
        idx = 0
        for n in ns:
            out.append(xs[idx : idx + int(n)])
            idx += int(n)
        return out
    return _ops.split_list_checked(*args, **kwargs)


def split_merge(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for split_merge.

    Args:
        *args: Positional arguments ``(predicate, xs)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2:
        pred = list(args[0])
        xs = list(args[1])
        lhs = [x for p, x in zip(pred, xs) if p]
        rhs = [x for p, x in zip(pred, xs) if not p]
        return lhs, rhs, lambda l_in, r_in: l_in + r_in
    return _ops.split_merge(*args, **kwargs)


def subvals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for subvals.

    Args:
        *args: Positional arguments ``(lst, replace)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2:
        lst = list(args[0])
        for idx, val in args[1]:
            lst[int(idx)] = val
        return tuple(lst)
    return _ops.subvals(*args, **kwargs)


def toposort(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for toposort.

    Args:
        *args: Positional arguments ``(end_nodes,)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if args:
        return list(args[0])
    return _ops.toposort(*args, **kwargs)


def unzip2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Unzip sequence of length-2 tuples into two tuples.

    Args:
        *args: Input parameter ``args`` (``xys``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        xs: list[ArrayLikeSpec] = []
        ys: list[ArrayLikeSpec] = []
        for a, b in args[0]:
            xs.append(a)
            ys.append(b)
        return tuple(xs), tuple(ys)
    return _ops.unzip2(*args, **kwargs)


def wrap_name(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for wrap_name.

    Args:
        *args: Positional arguments ``(name, transform_name)``.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    if len(args) >= 2:
        return f"{args[1]}({args[0]})"
    return _ops.wrap_name(*args, **kwargs)


def wraps(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Like functools.wraps, but with finer-grained control over the name and docstring.

    Args:
        *args: Input parameter ``args`` (``wrapped``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and callable(args[0]):
        return functools.wraps(args[0])
    return _ops.wraps(*args, **kwargs)
