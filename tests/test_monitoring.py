"""Tests for zero_jax.monitoring."""

from __future__ import annotations

import zero_jax.monitoring as mod


def test_clear_event_listeners() -> None:
    """Test clear_event_listeners."""
    events: list[str] = []
    mod.register_event_listener(lambda e: events.append(str(e)))
    mod.clear_event_listeners()
    mod.record_event("ignored")
    assert events == []


def test_record_event() -> None:
    """Test record_event."""
    events: list[str] = []
    mod.clear_event_listeners()
    mod.register_event_listener(lambda e: events.append(str(e)))
    mod.record_event("compile")
    assert events == ["compile"]
    mod.clear_event_listeners()


def test_record_event_duration_secs() -> None:
    """Test record_event_duration_secs."""
    durations: list[tuple[str, float]] = []
    mod.clear_event_listeners()
    mod.register_event_duration_secs_listener(lambda e, d: durations.append((str(e), float(d))))
    mod.record_event_duration_secs("jit", 0.25)
    assert durations == [("jit", 0.25)]
    mod.clear_event_listeners()


def test_register_event_duration_secs_listener() -> None:
    """Test register_event_duration_secs_listener."""
    durations: list[float] = []
    mod.clear_event_listeners()
    mod.register_event_duration_secs_listener(lambda _e, d: durations.append(float(d)))
    mod.record_event_duration_secs("trace", 1.5)
    assert durations == [1.5]
    mod.clear_event_listeners()


def test_register_event_listener() -> None:
    """Test register_event_listener."""
    events: list[str] = []
    mod.clear_event_listeners()
    mod.register_event_listener(lambda e: events.append(str(e)))
    mod.record_event("start")
    assert events == ["start"]
    mod.clear_event_listeners()
