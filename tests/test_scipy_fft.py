"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.fft as mod


def test_dct() -> None:
    """Test dct."""
    with patch("zero_jax._compiler_proxy_ops.dct", create=True) as mock_op:
        mod.dct()
        mock_op.assert_called_once_with()


def test_dctn() -> None:
    """Test dctn."""
    with patch("zero_jax._compiler_proxy_ops.dctn", create=True) as mock_op:
        mod.dctn()
        mock_op.assert_called_once_with()


def test_idct() -> None:
    """Test idct."""
    with patch("zero_jax._compiler_proxy_ops.idct", create=True) as mock_op:
        mod.idct()
        mock_op.assert_called_once_with()


def test_idctn() -> None:
    """Test idctn."""
    with patch("zero_jax._compiler_proxy_ops.idctn", create=True) as mock_op:
        mod.idctn()
        mock_op.assert_called_once_with()
