"""Tests for zero_jax module."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.scipy.ndimage as mod


def test_map_coordinates() -> None:
    """Test map_coordinates."""
    with patch("zero_jax._compiler_proxy_ops.map_coordinates", create=True) as mock_op:
        mod.map_coordinates()
        mock_op.assert_called_once_with()
