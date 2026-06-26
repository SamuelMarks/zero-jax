"""Frontend API routing for jax.monitoring."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def clear_event_listeners(*args: Any, **kwargs: Any) -> Any:
    """Clear event listeners."""
    return getattr(_ops, "clear_event_listeners")(*args, **kwargs)


def record_event(*args: Any, **kwargs: Any) -> Any:
    """Record an event."""
    return getattr(_ops, "record_event")(*args, **kwargs)


def record_event_duration_secs(*args: Any, **kwargs: Any) -> Any:
    """Record an event duration in seconds (float)."""
    return getattr(_ops, "record_event_duration_secs")(*args, **kwargs)


def register_event_duration_secs_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event_duration_secs()."""
    return getattr(_ops, "register_event_duration_secs_listener")(*args, **kwargs)


def register_event_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event()."""
    return getattr(_ops, "register_event_listener")(*args, **kwargs)
