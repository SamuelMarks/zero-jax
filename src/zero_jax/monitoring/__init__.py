"""Frontend API routing for jax.monitoring."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def clear_event_listeners(*args: Any, **kwargs: Any) -> Any:
    """Clear event listeners."""
    return _ops.clear_event_listeners(*args, **kwargs)


def record_event(*args: Any, **kwargs: Any) -> Any:
    """Record an event."""
    return _ops.record_event(*args, **kwargs)


def record_event_duration_secs(*args: Any, **kwargs: Any) -> Any:
    """Record an event duration in seconds (float)."""
    return _ops.record_event_duration_secs(*args, **kwargs)


def register_event_duration_secs_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event_duration_secs()."""
    return _ops.register_event_duration_secs_listener(*args, **kwargs)


def register_event_listener(*args: Any, **kwargs: Any) -> Any:
    """Register a callback to be invoked during record_event()."""
    return _ops.register_event_listener(*args, **kwargs)
