"""Tests for zero_jax module."""

import pytest
import zero_jax.ops as mod


def test_segment_max() -> None:
    """Test segment_max."""
    with pytest.raises(NotImplementedError):
        mod.segment_max()


def test_segment_min() -> None:
    """Test segment_min."""
    with pytest.raises(NotImplementedError):
        mod.segment_min()


def test_segment_prod() -> None:
    """Test segment_prod."""
    with pytest.raises(NotImplementedError):
        mod.segment_prod()


def test_segment_sum() -> None:
    """Test segment_sum."""
    with pytest.raises(NotImplementedError):
        mod.segment_sum()
