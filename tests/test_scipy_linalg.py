"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.linalg as mod


def test_block_diag() -> None:
    """Test block_diag."""
    with pytest.raises(NotImplementedError):
        mod.block_diag()


def test_cho_factor() -> None:
    """Test cho_factor."""
    with pytest.raises(NotImplementedError):
        mod.cho_factor()


def test_cho_solve() -> None:
    """Test cho_solve."""
    with pytest.raises(NotImplementedError):
        mod.cho_solve()


def test_cholesky() -> None:
    """Test cholesky."""
    with pytest.raises(NotImplementedError):
        mod.cholesky()


def test_det() -> None:
    """Test det."""
    with pytest.raises(NotImplementedError):
        mod.det()


def test_eigh() -> None:
    """Test eigh."""
    with pytest.raises(NotImplementedError):
        mod.eigh()


def test_eigh_tridiagonal() -> None:
    """Test eigh_tridiagonal."""
    with pytest.raises(NotImplementedError):
        mod.eigh_tridiagonal()


def test_expm() -> None:
    """Test expm."""
    with pytest.raises(NotImplementedError):
        mod.expm()


def test_expm_frechet() -> None:
    """Test expm_frechet."""
    with pytest.raises(NotImplementedError):
        mod.expm_frechet()


def test_funm() -> None:
    """Test funm."""
    with pytest.raises(NotImplementedError):
        mod.funm()


def test_hessenberg() -> None:
    """Test hessenberg."""
    with pytest.raises(NotImplementedError):
        mod.hessenberg()


def test_hilbert() -> None:
    """Test hilbert."""
    with pytest.raises(NotImplementedError):
        mod.hilbert()


def test_inv() -> None:
    """Test inv."""
    with pytest.raises(NotImplementedError):
        mod.inv()


def test_lu() -> None:
    """Test lu."""
    with pytest.raises(NotImplementedError):
        mod.lu()


def test_lu_factor() -> None:
    """Test lu_factor."""
    with pytest.raises(NotImplementedError):
        mod.lu_factor()


def test_lu_solve() -> None:
    """Test lu_solve."""
    with pytest.raises(NotImplementedError):
        mod.lu_solve()


def test_polar() -> None:
    """Test polar."""
    with pytest.raises(NotImplementedError):
        mod.polar()


def test_qr() -> None:
    """Test qr."""
    with pytest.raises(NotImplementedError):
        mod.qr()


def test_rsf2csf() -> None:
    """Test rsf2csf."""
    with pytest.raises(NotImplementedError):
        mod.rsf2csf()


def test_schur() -> None:
    """Test schur."""
    with pytest.raises(NotImplementedError):
        mod.schur()


def test_solve() -> None:
    """Test solve."""
    with pytest.raises(NotImplementedError):
        mod.solve()


def test_solve_triangular() -> None:
    """Test solve_triangular."""
    with pytest.raises(NotImplementedError):
        mod.solve_triangular()


def test_sqrtm() -> None:
    """Test sqrtm."""
    with pytest.raises(NotImplementedError):
        mod.sqrtm()


def test_svd() -> None:
    """Test svd."""
    with pytest.raises(NotImplementedError):
        mod.svd()


def test_toeplitz() -> None:
    """Test toeplitz."""
    with pytest.raises(NotImplementedError):
        mod.toeplitz()
