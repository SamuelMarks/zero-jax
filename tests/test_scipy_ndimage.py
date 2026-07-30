"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.ndimage as mod


def test_map_coordinates() -> None:
    """Test map_coordinates."""
    with patch("zero_jax._compiler_proxy_ops.map_coordinates", create=True) as mock_op:
        mod.map_coordinates()
        mock_op.assert_called_once_with()
