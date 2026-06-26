"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.ndimage as mod


def test_map_coordinates() -> None:
    """Test map_coordinates."""
    with patch("ml_switcheroo_compiler.ops.map_coordinates") as mock_op:
        mod.map_coordinates()
        mock_op.assert_called_once_with()
