"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.hardware_utils as mod


def test_num_available_tpu_chips_and_device_id() -> None:
    """Test num_available_tpu_chips_and_device_id."""
    with pytest.raises(NotImplementedError):
        mod.num_available_tpu_chips_and_device_id()


def test_tpu_enhanced_barrier_supported() -> None:
    """Test tpu_enhanced_barrier_supported."""
    with pytest.raises(NotImplementedError):
        mod.tpu_enhanced_barrier_supported()
