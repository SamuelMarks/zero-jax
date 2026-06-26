"""Tests for zero_jax module."""

import pytest
import zero_jax.custom_transpose as mod


def test_custom_transpose() -> None:
    """Test custom_transpose."""
    obj = mod.custom_transpose()
    assert obj is not None
