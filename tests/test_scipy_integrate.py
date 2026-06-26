"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.integrate as mod


def test_trapezoid() -> None:
    """Test trapezoid."""
    with patch("ml_switcheroo_compiler.ops.trapezoid") as mock_op:
        mod.trapezoid()
        mock_op.assert_called_once_with()
