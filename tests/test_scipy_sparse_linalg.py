"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.sparse.linalg as mod


def test_bicgstab() -> None:
    """Test bicgstab."""
    with patch("zero_jax._compiler_proxy_ops.bicgstab", create=True) as mock_op:
        mod.bicgstab()
        mock_op.assert_called_once_with()


def test_cg() -> None:
    """Test cg."""
    with patch("zero_jax._compiler_proxy_ops.cg", create=True) as mock_op:
        mod.cg()
        mock_op.assert_called_once_with()


def test_gmres() -> None:
    """Test gmres."""
    with patch("zero_jax._compiler_proxy_ops.gmres", create=True) as mock_op:
        mod.gmres()
        mock_op.assert_called_once_with()
