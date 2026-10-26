"""Tests for zero_jax.lax.linalg."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.numpy as jnp
from tests.conftest import assert_allclose
from zero_jax.lax import linalg


def test_cholesky() -> None:
    """Test cholesky."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    assert_allclose(linalg.cholesky(a), [[2.0, 0.0], [0.0, 3.0]])


def test_cholesky_p() -> None:
    """Test cholesky_p."""
    assert linalg.cholesky_p.name == "cholesky"


def test_eig() -> None:
    """Test eig."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    res = linalg.eig(a)
    assert isinstance(res, tuple)
    assert linalg.eig(a, compute_v=False).shape == (2,)
    with patch("zero_jax._compiler_proxy_ops.eig", return_value=jnp.array([2.0, 3.0])._tensor):
        assert_allclose(linalg.eig(a), [2.0, 3.0])


def test_eig_p() -> None:
    """Test eig_p."""
    assert linalg.eig_p.name == "eig"


def test_eigh() -> None:
    """Test eigh."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    res = linalg.eigh(a)
    assert isinstance(res, tuple)
    with patch("zero_jax._compiler_proxy_ops.eigh", return_value=jnp.array([2.0, 3.0])._tensor):
        assert_allclose(linalg.eigh(a), [2.0, 3.0])


def test_eigh_p() -> None:
    """Test eigh_p."""
    assert linalg.eigh_p.name == "eigh"


def test_hessenberg() -> None:
    """Test hessenberg."""
    a = jnp.array([[2.0, 1.0], [0.0, 3.0]])
    h, q = linalg.hessenberg(a)
    assert_allclose(h, a)
    assert_allclose(q, jnp.eye(2))
    with patch("zero_jax._compiler_proxy_ops.hessenberg", return_value=a._tensor):
        assert_allclose(linalg.hessenberg(a), a)


def test_hessenberg_p() -> None:
    """Test hessenberg_p."""
    assert linalg.hessenberg_p.name == "hessenberg"


def test_householder_product() -> None:
    """Test householder_product."""
    a = jnp.eye(2)
    taus = jnp.zeros((2,))
    with patch("zero_jax._compiler_proxy_ops.householder_product", return_value=a._tensor):
        assert_allclose(linalg.householder_product(a, taus), a)


def test_householder_product_p() -> None:
    """Test householder_product_p."""
    assert linalg.householder_product_p.name == "householder_product"


def test_lu() -> None:
    """Test lu."""
    a = jnp.eye(2)
    res = linalg.lu(a)
    assert isinstance(res, tuple)
    with patch("zero_jax._compiler_proxy_ops.lu", return_value=a._tensor):
        assert_allclose(linalg.lu(a), a)


def test_lu_p() -> None:
    """Test lu_p."""
    assert linalg.lu_p.name == "lu"


def test_lu_pivots_to_permutation() -> None:
    """Test lu_pivots_to_permutation."""
    piv = jnp.array([0, 1])
    with patch("zero_jax._compiler_proxy_ops.lu_pivots_to_permutation", return_value=piv._tensor):
        assert_allclose(linalg.lu_pivots_to_permutation(piv, 2), [0, 1])


def test_qdwh() -> None:
    """Test qdwh."""
    a = jnp.eye(2)
    with patch("zero_jax._compiler_proxy_ops.qdwh", return_value=(a._tensor, a._tensor)):
        u, h = linalg.qdwh(a)
        assert_allclose(u, a)
        assert_allclose(h, a)
    with patch("zero_jax._compiler_proxy_ops.qdwh", return_value=a._tensor):
        assert_allclose(linalg.qdwh(a), a)


def test_qr() -> None:
    """Test qr."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    q, r = linalg.qr(a)
    assert_allclose(q @ r, a)
    with patch("zero_jax._compiler_proxy_ops.qr", return_value=a._tensor):
        assert_allclose(linalg.qr(a), a)


def test_qr_p() -> None:
    """Test qr_p."""
    assert linalg.qr_p.name == "qr"


def test_schur() -> None:
    """Test schur."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    t_mat, q_mat = linalg.schur(a)
    assert_allclose(q_mat @ t_mat @ jnp.transpose(q_mat), a)
    with patch("zero_jax._compiler_proxy_ops.schur", return_value=a._tensor):
        assert_allclose(linalg.schur(a), a)


def test_schur_p() -> None:
    """Test schur_p."""
    assert linalg.schur_p.name == "schur"


def test_svd() -> None:
    """Test svd."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    u, s, vh = linalg.svd(a)
    assert_allclose(u @ jnp.diag(s) @ vh, a)
    s_only = linalg.svd(a, compute_uv=False)
    assert_allclose(jnp.sort(s_only), [2.0, 3.0])


def test_svd_p() -> None:
    """Test svd_p."""
    assert linalg.svd_p.name == "svd"


def test_triangular_solve() -> None:
    """Test triangular_solve."""
    a = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    b = jnp.array([[6.0], [8.0]])
    with patch("zero_jax._compiler_proxy_ops.triangular_solve", return_value=jnp.array([[3.0], [2.0]])._tensor):
        assert_allclose(linalg.triangular_solve(a, b, left_side=True, lower=True), [[3.0], [2.0]])


def test_triangular_solve_p() -> None:
    """Test triangular_solve_p."""
    assert linalg.triangular_solve_p.name == "triangular_solve"


def test_tridiagonal() -> None:
    """Test tridiagonal."""
    a = jnp.eye(2)
    with patch("zero_jax._compiler_proxy_ops.tridiagonal", return_value=(a._tensor, a._tensor)):
        t1, t2 = linalg.tridiagonal(a)
        assert_allclose(t1, a)
        assert_allclose(t2, a)
    with patch("zero_jax._compiler_proxy_ops.tridiagonal", return_value=a._tensor):
        assert_allclose(linalg.tridiagonal(a), a)


def test_tridiagonal_p() -> None:
    """Test tridiagonal_p."""
    assert linalg.tridiagonal_p.name == "tridiagonal"


def test_tridiagonal_solve() -> None:
    """Test tridiagonal_solve."""
    dl = jnp.array([0.0, 0.0])
    d = jnp.array([2.0, 4.0])
    du = jnp.array([0.0, 0.0])
    b = jnp.array([[6.0], [8.0]])
    with patch("zero_jax._compiler_proxy_ops.tridiagonal_solve", return_value=jnp.array([[3.0], [2.0]])._tensor):
        assert_allclose(linalg.tridiagonal_solve(dl, d, du, b), [[3.0], [2.0]])


def test_tridiagonal_solve_p() -> None:
    """Test tridiagonal_solve_p."""
    assert linalg.tridiagonal_solve_p.name == "tridiagonal_solve"
