"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.fft as mod


def test_dct() -> None:
    """Test dct."""
    with patch("ml_switcheroo_compiler.ops.dct") as mock_op:
        mod.dct()
        mock_op.assert_called_once_with()


def test_dctn() -> None:
    """Test dctn."""
    with patch("ml_switcheroo_compiler.ops.dctn") as mock_op:
        mod.dctn()
        mock_op.assert_called_once_with()


def test_idct() -> None:
    """Test idct."""
    with patch("ml_switcheroo_compiler.ops.idct") as mock_op:
        mod.idct()
        mock_op.assert_called_once_with()


def test_idctn() -> None:
    """Test idctn."""
    with patch("ml_switcheroo_compiler.ops.idctn") as mock_op:
        mod.idctn()
        mock_op.assert_called_once_with()
