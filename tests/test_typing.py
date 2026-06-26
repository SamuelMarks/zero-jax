"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.typing as mod


def test_ArrayLike() -> None:
    """Test ArrayLike."""
    with patch("ml_switcheroo_compiler.ops.ArrayLike") as mock_op:
        mod.ArrayLike()
        mock_op.assert_called_once_with()


def test_DTypeLike() -> None:
    """Test DTypeLike."""
    with patch("ml_switcheroo_compiler.ops.DTypeLike") as mock_op:
        mod.DTypeLike()
        mock_op.assert_called_once_with()
