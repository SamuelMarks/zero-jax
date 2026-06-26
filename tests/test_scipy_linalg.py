"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.linalg as mod


def test_block_diag() -> None:
    """Test block_diag."""
    with patch("ml_switcheroo_compiler.ops.block_diag") as mock_op:
        mod.block_diag()
        mock_op.assert_called_once_with()


def test_cho_factor() -> None:
    """Test cho_factor."""
    with patch("ml_switcheroo_compiler.ops.cho_factor") as mock_op:
        mod.cho_factor()
        mock_op.assert_called_once_with()


def test_cho_solve() -> None:
    """Test cho_solve."""
    with patch("ml_switcheroo_compiler.ops.cho_solve") as mock_op:
        mod.cho_solve()
        mock_op.assert_called_once_with()


def test_cholesky() -> None:
    """Test cholesky."""
    with patch("ml_switcheroo_compiler.ops.cholesky") as mock_op:
        mod.cholesky()
        mock_op.assert_called_once_with()


def test_det() -> None:
    """Test det."""
    with patch("ml_switcheroo_compiler.ops.det") as mock_op:
        mod.det()
        mock_op.assert_called_once_with()


def test_eigh() -> None:
    """Test eigh."""
    with patch("ml_switcheroo_compiler.ops.eigh") as mock_op:
        mod.eigh()
        mock_op.assert_called_once_with()


def test_eigh_tridiagonal() -> None:
    """Test eigh_tridiagonal."""
    with patch("ml_switcheroo_compiler.ops.eigh_tridiagonal") as mock_op:
        mod.eigh_tridiagonal()
        mock_op.assert_called_once_with()


def test_expm() -> None:
    """Test expm."""
    with patch("ml_switcheroo_compiler.ops.expm") as mock_op:
        mod.expm()
        mock_op.assert_called_once_with()


def test_expm_frechet() -> None:
    """Test expm_frechet."""
    with patch("ml_switcheroo_compiler.ops.expm_frechet") as mock_op:
        mod.expm_frechet()
        mock_op.assert_called_once_with()


def test_funm() -> None:
    """Test funm."""
    with patch("ml_switcheroo_compiler.ops.funm") as mock_op:
        mod.funm()
        mock_op.assert_called_once_with()


def test_hessenberg() -> None:
    """Test hessenberg."""
    with patch("ml_switcheroo_compiler.ops.hessenberg") as mock_op:
        mod.hessenberg()
        mock_op.assert_called_once_with()


def test_hilbert() -> None:
    """Test hilbert."""
    with patch("ml_switcheroo_compiler.ops.hilbert") as mock_op:
        mod.hilbert()
        mock_op.assert_called_once_with()


def test_inv() -> None:
    """Test inv."""
    with patch("ml_switcheroo_compiler.ops.inv") as mock_op:
        mod.inv()
        mock_op.assert_called_once_with()


def test_lu() -> None:
    """Test lu."""
    with patch("ml_switcheroo_compiler.ops.lu") as mock_op:
        mod.lu()
        mock_op.assert_called_once_with()


def test_lu_factor() -> None:
    """Test lu_factor."""
    with patch("ml_switcheroo_compiler.ops.lu_factor") as mock_op:
        mod.lu_factor()
        mock_op.assert_called_once_with()


def test_lu_solve() -> None:
    """Test lu_solve."""
    with patch("ml_switcheroo_compiler.ops.lu_solve") as mock_op:
        mod.lu_solve()
        mock_op.assert_called_once_with()


def test_polar() -> None:
    """Test polar."""
    with patch("ml_switcheroo_compiler.ops.polar") as mock_op:
        mod.polar()
        mock_op.assert_called_once_with()


def test_qr() -> None:
    """Test qr."""
    with patch("ml_switcheroo_compiler.ops.qr") as mock_op:
        mod.qr()
        mock_op.assert_called_once_with()


def test_rsf2csf() -> None:
    """Test rsf2csf."""
    with patch("ml_switcheroo_compiler.ops.rsf2csf") as mock_op:
        mod.rsf2csf()
        mock_op.assert_called_once_with()


def test_schur() -> None:
    """Test schur."""
    with patch("ml_switcheroo_compiler.ops.schur") as mock_op:
        mod.schur()
        mock_op.assert_called_once_with()


def test_solve() -> None:
    """Test solve."""
    with patch("ml_switcheroo_compiler.ops.solve") as mock_op:
        mod.solve()
        mock_op.assert_called_once_with()


def test_solve_triangular() -> None:
    """Test solve_triangular."""
    with patch("ml_switcheroo_compiler.ops.solve_triangular") as mock_op:
        mod.solve_triangular()
        mock_op.assert_called_once_with()


def test_sqrtm() -> None:
    """Test sqrtm."""
    with patch("ml_switcheroo_compiler.ops.sqrtm") as mock_op:
        mod.sqrtm()
        mock_op.assert_called_once_with()


def test_svd() -> None:
    """Test svd."""
    with patch("ml_switcheroo_compiler.ops.svd") as mock_op:
        mod.svd()
        mock_op.assert_called_once_with()


def test_toeplitz() -> None:
    """Test toeplitz."""
    with patch("ml_switcheroo_compiler.ops.toeplitz") as mock_op:
        mod.toeplitz()
        mock_op.assert_called_once_with()
