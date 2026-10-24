"""Tests for zero_jax module."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.scipy.integrate as mod


def test_trapezoid() -> None:
    """Test trapezoid."""
    with patch("zero_jax._compiler_proxy_ops.trapezoid", create=True) as mock_op:
        mod.trapezoid()
        mock_op.assert_called_once_with()
