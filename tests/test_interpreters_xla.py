"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.xla as mod


def test_Backend() -> None:
    """Test Backend."""
    obj = mod.Backend()
    assert obj is not None


def test_abstractify() -> None:
    """Test abstractify."""
    with patch("ml_switcheroo_compiler.ops.abstractify") as mock_op:
        mod.abstractify()
        mock_op.assert_called_once_with()


def test_apply_primitive() -> None:
    """Test apply_primitive."""
    with patch("ml_switcheroo_compiler.ops.apply_primitive") as mock_op:
        mod.apply_primitive()
        mock_op.assert_called_once_with()


def test_canonicalize_dtype() -> None:
    """Test canonicalize_dtype."""
    with patch("ml_switcheroo_compiler.ops.canonicalize_dtype") as mock_op:
        mod.canonicalize_dtype()
        mock_op.assert_called_once_with()
