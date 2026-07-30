"""Tests for zero_jax.lax.linalg."""

from unittest.mock import patch

import pytest

from zero_jax.lax import linalg


def test_cholesky() -> None:
    """Test cholesky."""
    with patch("zero_jax._compiler_proxy_ops.cholesky", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert linalg.cholesky(1.0) is not None


def test_cholesky_p() -> None:
    """Test cholesky_p."""
    assert linalg.cholesky_p is not None


def test_eig() -> None:
    """Test eig."""
    with patch("zero_jax._compiler_proxy_ops.eig", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.eig(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.eig(1.0) is not None


def test_eig_p() -> None:
    """Test eig_p."""
    assert linalg.eig_p is not None


def test_eigh() -> None:
    """Test eigh."""
    with patch("zero_jax._compiler_proxy_ops.eigh", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.eigh(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.eigh(1.0) is not None


def test_eigh_p() -> None:
    """Test eigh_p."""
    assert linalg.eigh_p is not None


def test_hessenberg() -> None:
    """Test hessenberg."""
    with patch("zero_jax._compiler_proxy_ops.hessenberg", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.hessenberg(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.hessenberg(1.0) is not None


def test_hessenberg_p() -> None:
    """Test hessenberg_p."""
    assert linalg.hessenberg_p is not None


def test_householder_product() -> None:
    """Test householder_product."""
    with patch(
        "zero_jax._compiler_proxy_ops.householder_product", create=True
    ) as mock_op:
        mock_op.return_value = 1.0
        assert linalg.householder_product(1.0, 1.0) is not None


def test_householder_product_p() -> None:
    """Test householder_product_p."""
    assert linalg.householder_product_p is not None


def test_lu() -> None:
    """Test lu."""
    with patch("zero_jax._compiler_proxy_ops.lu", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.lu(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.lu(1.0) is not None


def test_lu_p() -> None:
    """Test lu_p."""
    assert linalg.lu_p is not None


def test_lu_pivots_to_permutation() -> None:
    """Test lu_pivots_to_permutation."""
    with patch(
        "zero_jax._compiler_proxy_ops.lu_pivots_to_permutation", create=True
    ) as mock_op:
        mock_op.return_value = 1.0
        assert linalg.lu_pivots_to_permutation(1.0, 1) is not None


def test_qdwh() -> None:
    """Test qdwh."""
    with patch("zero_jax._compiler_proxy_ops.qdwh", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.qdwh(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.qdwh(1.0) is not None


def test_qr() -> None:
    """Test qr."""
    with patch("zero_jax._compiler_proxy_ops.qr", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.qr(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.qr(1.0) is not None


def test_qr_p() -> None:
    """Test qr_p."""
    assert linalg.qr_p is not None


def test_schur() -> None:
    """Test schur."""
    with patch("zero_jax._compiler_proxy_ops.schur", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.schur(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.schur(1.0) is not None


def test_schur_p() -> None:
    """Test schur_p."""
    assert linalg.schur_p is not None


def test_svd() -> None:
    """Test svd."""
    with patch("zero_jax._compiler_proxy_ops.svd", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.svd(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.svd(1.0) is not None


def test_svd_p() -> None:
    """Test svd_p."""
    assert linalg.svd_p is not None


def test_triangular_solve() -> None:
    """Test triangular_solve."""
    with patch("zero_jax._compiler_proxy_ops.triangular_solve", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert linalg.triangular_solve(1.0, 1.0) is not None


def test_triangular_solve_p() -> None:
    """Test triangular_solve_p."""
    assert linalg.triangular_solve_p is not None


def test_tridiagonal() -> None:
    """Test tridiagonal."""
    with patch("zero_jax._compiler_proxy_ops.tridiagonal", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert linalg.tridiagonal(1.0) is not None
        mock_op.return_value = 1.0
        assert linalg.tridiagonal(1.0) is not None


def test_tridiagonal_p() -> None:
    """Test tridiagonal_p."""
    assert linalg.tridiagonal_p is not None


def test_tridiagonal_solve() -> None:
    """Test tridiagonal_solve."""
    with patch(
        "zero_jax._compiler_proxy_ops.tridiagonal_solve", create=True
    ) as mock_op:
        mock_op.return_value = 1.0
        assert linalg.tridiagonal_solve(1.0, 1.0, 1.0, 1.0) is not None


def test_tridiagonal_solve_p() -> None:
    """Test tridiagonal_solve_p."""
    assert linalg.tridiagonal_solve_p is not None
