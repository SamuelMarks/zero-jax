"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.ndimage as mod


def test_map_coordinates() -> None:
    """Test map_coordinates."""
    with pytest.raises(NotImplementedError):
        mod.map_coordinates()
