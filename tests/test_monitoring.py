"""Tests for zero_jax module."""

import pytest
import zero_jax.monitoring as mod


def test_clear_event_listeners() -> None:
    """Test clear_event_listeners."""
    with pytest.raises(NotImplementedError):
        mod.clear_event_listeners()


def test_record_event() -> None:
    """Test record_event."""
    with pytest.raises(NotImplementedError):
        mod.record_event()


def test_record_event_duration_secs() -> None:
    """Test record_event_duration_secs."""
    with pytest.raises(NotImplementedError):
        mod.record_event_duration_secs()


def test_register_event_duration_secs_listener() -> None:
    """Test register_event_duration_secs_listener."""
    with pytest.raises(NotImplementedError):
        mod.register_event_duration_secs_listener()


def test_register_event_listener() -> None:
    """Test register_event_listener."""
    with pytest.raises(NotImplementedError):
        mod.register_event_listener()
