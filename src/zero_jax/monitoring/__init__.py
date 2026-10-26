"""Frontend API routing for jax.monitoring."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable

_EVENT_LISTENERS: list[VariadicCallable[ConfigValue]] = []
_DURATION_LISTENERS: list[VariadicCallable[ConfigValue]] = []


def clear_event_listeners() -> None:
    """Clear all registered event and duration listeners."""
    _EVENT_LISTENERS.clear()
    _DURATION_LISTENERS.clear()


def record_event(event: str, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> None:
    """Record a named monitoring event.

    Args:
        event: Event name string.
        *args: Additional positional event arguments.
        **kwargs: Additional keyword event arguments.
    """
    for cb in _EVENT_LISTENERS:
        cb(event, *args, **kwargs)


def record_event_duration_secs(event: str, duration: float, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> None:
    """Record an event duration in seconds.

    Args:
        event: Event name string.
        duration: Elapsed duration in seconds.
        *args: Additional positional event arguments.
        **kwargs: Additional keyword event arguments.
    """
    for cb in _DURATION_LISTENERS:
        cb(event, duration, *args, **kwargs)


def register_event_duration_secs_listener(callback: VariadicCallable[ConfigValue]) -> None:
    """Register a callback to be invoked during :func:`record_event_duration_secs`.

    Args:
        callback: Listener callable accepting ``(event, duration, *args, **kwargs)``.
    """
    _DURATION_LISTENERS.append(callback)


def register_event_listener(callback: VariadicCallable[ConfigValue]) -> None:
    """Register a callback to be invoked during :func:`record_event`.

    Args:
        callback: Listener callable accepting ``(event, *args, **kwargs)``.
    """
    _EVENT_LISTENERS.append(callback)


__all__ = [
    "clear_event_listeners",
    "record_event",
    "record_event_duration_secs",
    "register_event_duration_secs_listener",
    "register_event_listener",
]
