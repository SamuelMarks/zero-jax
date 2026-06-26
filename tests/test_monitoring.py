"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.monitoring as mod


def test_clear_event_listeners() -> None:
    """Test clear_event_listeners."""
    with patch("ml_switcheroo_compiler.ops.clear_event_listeners") as mock_op:
        mod.clear_event_listeners()
        mock_op.assert_called_once_with()


def test_record_event() -> None:
    """Test record_event."""
    with patch("ml_switcheroo_compiler.ops.record_event") as mock_op:
        mod.record_event()
        mock_op.assert_called_once_with()


def test_record_event_duration_secs() -> None:
    """Test record_event_duration_secs."""
    with patch("ml_switcheroo_compiler.ops.record_event_duration_secs") as mock_op:
        mod.record_event_duration_secs()
        mock_op.assert_called_once_with()


def test_register_event_duration_secs_listener() -> None:
    """Test register_event_duration_secs_listener."""
    with patch(
        "ml_switcheroo_compiler.ops.register_event_duration_secs_listener"
    ) as mock_op:
        mod.register_event_duration_secs_listener()
        mock_op.assert_called_once_with()


def test_register_event_listener() -> None:
    """Test register_event_listener."""
    with patch("ml_switcheroo_compiler.ops.register_event_listener") as mock_op:
        mod.register_event_listener()
        mock_op.assert_called_once_with()
