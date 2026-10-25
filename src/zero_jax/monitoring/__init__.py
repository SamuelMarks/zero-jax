"""Frontend API routing for jax.monitoring."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable

_EVENT_LISTENERS: list[VariadicCallable[ConfigValue]] = []
_DURATION_LISTENERS: list[VariadicCallable[ConfigValue]] = []


def clear_event_listeners(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Clear event listeners.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    _EVENT_LISTENERS.clear()
    _DURATION_LISTENERS.clear()
    if not args and not kwargs:
        try:
            return _ops.clear_event_listeners()
        except AttributeError:
            return None
    return None


def record_event(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Record an event.

    Args:
        *args: Input parameter ``args`` (``event``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        for cb in _EVENT_LISTENERS:
            cb(*args, **kwargs)
        return None
    return _ops.record_event(*args, **kwargs)


def record_event_duration_secs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Record an event duration in seconds (float).

    Args:
        *args: Input parameter ``args`` (``event``, ``duration``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        for cb in _DURATION_LISTENERS:
            cb(*args, **kwargs)
        return None
    return _ops.record_event_duration_secs(*args, **kwargs)


def register_event_duration_secs_listener(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Register a callback to be invoked during record_event_duration_secs().

    Args:
        *args: Input parameter ``args`` (``callback``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and callable(args[0]):
        _DURATION_LISTENERS.append(args[0])
        return None
    return _ops.register_event_duration_secs_listener(*args, **kwargs)


def register_event_listener(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Register a callback to be invoked during record_event().

    Args:
        *args: Input parameter ``args`` (``callback``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and callable(args[0]):
        _EVENT_LISTENERS.append(args[0])
        return None
    return _ops.register_event_listener(*args, **kwargs)
