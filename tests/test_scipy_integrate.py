"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.integrate as mod


def test_trapezoid() -> None:
    """Test trapezoid."""
    with pytest.raises(NotImplementedError):
        mod.trapezoid()
