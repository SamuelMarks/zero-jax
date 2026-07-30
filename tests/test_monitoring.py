"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.monitoring as mod


def test_clear_event_listeners() -> None:
    """Test clear_event_listeners."""
    with patch(
        "zero_jax._compiler_proxy_ops.clear_event_listeners", create=True
    ) as mock_op:
        mod.clear_event_listeners()
        mock_op.assert_called_once_with()


def test_record_event() -> None:
    """Test record_event."""
    with patch("zero_jax._compiler_proxy_ops.record_event", create=True) as mock_op:
        mod.record_event()
        mock_op.assert_called_once_with()


def test_record_event_duration_secs() -> None:
    """Test record_event_duration_secs."""
    with patch(
        "zero_jax._compiler_proxy_ops.record_event_duration_secs", create=True
    ) as mock_op:
        mod.record_event_duration_secs()
        mock_op.assert_called_once_with()


def test_register_event_duration_secs_listener() -> None:
    """Test register_event_duration_secs_listener."""
    with patch(
        "zero_jax._compiler_proxy_ops.register_event_duration_secs_listener",
        create=True,
    ) as mock_op:
        mod.register_event_duration_secs_listener()
        mock_op.assert_called_once_with()


def test_register_event_listener() -> None:
    """Test register_event_listener."""
    with patch(
        "zero_jax._compiler_proxy_ops.register_event_listener", create=True
    ) as mock_op:
        mod.register_event_listener()
        mock_op.assert_called_once_with()
