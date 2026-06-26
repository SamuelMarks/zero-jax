"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.sparse.linalg as mod


def test_bicgstab() -> None:
    """Test bicgstab."""
    with patch("ml_switcheroo_compiler.ops.bicgstab") as mock_op:
        mod.bicgstab()
        mock_op.assert_called_once_with()


def test_cg() -> None:
    """Test cg."""
    with patch("ml_switcheroo_compiler.ops.cg") as mock_op:
        mod.cg()
        mock_op.assert_called_once_with()


def test_gmres() -> None:
    """Test gmres."""
    with patch("ml_switcheroo_compiler.ops.gmres") as mock_op:
        mod.gmres()
        mock_op.assert_called_once_with()
