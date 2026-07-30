"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.linalg as mod


def test_block_diag() -> None:
    """Test block_diag."""
    with patch("zero_jax._compiler_proxy_ops.block_diag", create=True) as mock_op:
        mod.block_diag()
        mock_op.assert_called_once_with()


def test_cho_factor() -> None:
    """Test cho_factor."""
    with patch("zero_jax._compiler_proxy_ops.cho_factor", create=True) as mock_op:
        mod.cho_factor()
        mock_op.assert_called_once_with()


def test_cho_solve() -> None:
    """Test cho_solve."""
    with patch("zero_jax._compiler_proxy_ops.cho_solve", create=True) as mock_op:
        mod.cho_solve()
        mock_op.assert_called_once_with()


def test_cholesky() -> None:
    """Test cholesky."""
    with patch("zero_jax._compiler_proxy_ops.cholesky", create=True) as mock_op:
        mod.cholesky()
        mock_op.assert_called_once_with()


def test_det() -> None:
    """Test det."""
    with patch("zero_jax._compiler_proxy_ops.det", create=True) as mock_op:
        mod.det()
        mock_op.assert_called_once_with()


def test_eigh() -> None:
    """Test eigh."""
    with patch("zero_jax._compiler_proxy_ops.eigh", create=True) as mock_op:
        mod.eigh()
        mock_op.assert_called_once_with()


def test_eigh_tridiagonal() -> None:
    """Test eigh_tridiagonal."""
    with patch("zero_jax._compiler_proxy_ops.eigh_tridiagonal", create=True) as mock_op:
        mod.eigh_tridiagonal()
        mock_op.assert_called_once_with()


def test_expm() -> None:
    """Test expm."""
    with patch("zero_jax._compiler_proxy_ops.expm", create=True) as mock_op:
        mod.expm()
        mock_op.assert_called_once_with()


def test_expm_frechet() -> None:
    """Test expm_frechet."""
    with patch("zero_jax._compiler_proxy_ops.expm_frechet", create=True) as mock_op:
        mod.expm_frechet()
        mock_op.assert_called_once_with()


def test_funm() -> None:
    """Test funm."""
    with patch("zero_jax._compiler_proxy_ops.funm", create=True) as mock_op:
        mod.funm()
        mock_op.assert_called_once_with()


def test_hessenberg() -> None:
    """Test hessenberg."""
    with patch("zero_jax._compiler_proxy_ops.hessenberg", create=True) as mock_op:
        mod.hessenberg()
        mock_op.assert_called_once_with()


def test_hilbert() -> None:
    """Test hilbert."""
    with patch("zero_jax._compiler_proxy_ops.hilbert", create=True) as mock_op:
        mod.hilbert()
        mock_op.assert_called_once_with()


def test_inv() -> None:
    """Test inv."""
    with patch("zero_jax._compiler_proxy_ops.inv", create=True) as mock_op:
        mod.inv()
        mock_op.assert_called_once_with()


def test_lu() -> None:
    """Test lu."""
    with patch("zero_jax._compiler_proxy_ops.lu", create=True) as mock_op:
        mod.lu()
        mock_op.assert_called_once_with()


def test_lu_factor() -> None:
    """Test lu_factor."""
    with patch("zero_jax._compiler_proxy_ops.lu_factor", create=True) as mock_op:
        mod.lu_factor()
        mock_op.assert_called_once_with()


def test_lu_solve() -> None:
    """Test lu_solve."""
    with patch("zero_jax._compiler_proxy_ops.lu_solve", create=True) as mock_op:
        mod.lu_solve()
        mock_op.assert_called_once_with()


def test_polar() -> None:
    """Test polar."""
    with patch("zero_jax._compiler_proxy_ops.polar", create=True) as mock_op:
        mod.polar()
        mock_op.assert_called_once_with()


def test_qr() -> None:
    """Test qr."""
    with patch("zero_jax._compiler_proxy_ops.qr", create=True) as mock_op:
        mod.qr()
        mock_op.assert_called_once_with()


def test_rsf2csf() -> None:
    """Test rsf2csf."""
    with patch("zero_jax._compiler_proxy_ops.rsf2csf", create=True) as mock_op:
        mod.rsf2csf()
        mock_op.assert_called_once_with()


def test_schur() -> None:
    """Test schur."""
    with patch("zero_jax._compiler_proxy_ops.schur", create=True) as mock_op:
        mod.schur()
        mock_op.assert_called_once_with()


def test_solve() -> None:
    """Test solve."""
    with patch("zero_jax._compiler_proxy_ops.solve", create=True) as mock_op:
        mod.solve()
        mock_op.assert_called_once_with()


def test_solve_triangular() -> None:
    """Test solve_triangular."""
    with patch("zero_jax._compiler_proxy_ops.solve_triangular", create=True) as mock_op:
        mod.solve_triangular()
        mock_op.assert_called_once_with()


def test_sqrtm() -> None:
    """Test sqrtm."""
    with patch("zero_jax._compiler_proxy_ops.sqrtm", create=True) as mock_op:
        mod.sqrtm()
        mock_op.assert_called_once_with()


def test_svd() -> None:
    """Test svd."""
    with patch("zero_jax._compiler_proxy_ops.svd", create=True) as mock_op:
        mod.svd()
        mock_op.assert_called_once_with()


def test_toeplitz() -> None:
    """Test toeplitz."""
    with patch("zero_jax._compiler_proxy_ops.toeplitz", create=True) as mock_op:
        mod.toeplitz()
        mock_op.assert_called_once_with()
