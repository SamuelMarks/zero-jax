"""Frontend API routing for jax.monitoring."""

from typing import Any


def clear_event_listeners(*args: Any, **kwargs: Any) -> Any:
    """Clear event listeners."""
    raise NotImplementedError("clear_event_listeners not yet implemented in zero-jax")


def record_event(*args: Any, **kwargs: Any) -> Any:
    """Record an event."""
    raise NotImplementedError("record_event not yet implemented in zero-jax")


def record_event_duration_secs(*args: Any, **kwargs: Any) -> Any:
    """Record an event duration in seconds (float)."""
    raise NotImplementedError(
        "record_event_duration_secs not yet implemented in zero-jax"
    )


def register_event_duration_secs_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event_duration_secs()."""
    raise NotImplementedError(
        "register_event_duration_secs_listener not yet implemented in zero-jax"
    )


def register_event_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event()."""
    raise NotImplementedError("register_event_listener not yet implemented in zero-jax")
