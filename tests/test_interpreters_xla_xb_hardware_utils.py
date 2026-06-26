"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.xla.xb.hardware_utils as mod


def test_num_available_tpu_chips_and_device_id() -> None:
    """Test num_available_tpu_chips_and_device_id."""
    with patch(
        "ml_switcheroo_compiler.ops.num_available_tpu_chips_and_device_id"
    ) as mock_op:
        mod.num_available_tpu_chips_and_device_id()
        mock_op.assert_called_once_with()


def test_tpu_enhanced_barrier_supported() -> None:
    """Test tpu_enhanced_barrier_supported."""
    with patch("ml_switcheroo_compiler.ops.tpu_enhanced_barrier_supported") as mock_op:
        mod.tpu_enhanced_barrier_supported()
        mock_op.assert_called_once_with()
