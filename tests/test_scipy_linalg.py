"""Tests for zero_jax.scipy.linalg."""

from __future__ import annotations

import pytest

import zero_jax.numpy as jnp
import zero_jax.scipy.linalg as mod
from tests.conftest import assert_allclose


def test_block_diag() -> None:
    """Test block_diag."""
    assert mod.block_diag().shape == (0, 0)
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0]])
    res = mod.block_diag(a, b)
    assert_allclose(res, [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])


def test_cho_factor() -> None:
    """Test cho_factor."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    c, lower = mod.cho_factor(a, lower=True)
    assert lower is True
    assert_allclose(c, [[2.0, 0.0], [0.0, 3.0]])
    c_int, lower_int = mod.cho_factor(jnp.array([[4, 0], [0, 9]]), lower=False)
    assert lower_int is False
    assert_allclose(c_int, [[2.0, 0.0], [0.0, 3.0]])


def test_cho_solve() -> None:
    """Test cho_solve."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    b = jnp.array([8.0, 18.0])
    cf_low = mod.cho_factor(a, lower=True)
    assert_allclose(mod.cho_solve(cf_low, b), [2.0, 2.0])
    cf_up = mod.cho_factor(a, lower=False)
    assert_allclose(mod.cho_solve(cf_up, b), [2.0, 2.0])


def test_cholesky() -> None:
    """Test cholesky."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    assert_allclose(mod.cholesky(a, lower=True), [[2.0, 0.0], [0.0, 3.0]])


def test_det() -> None:
    """Test det."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    assert_allclose(mod.det(a), 6.0)


def test_eigh() -> None:
    """Test eigh."""
    a = jnp.array([[2.0, 0.0], [0.0, 5.0]])
    w, v = mod.eigh(a)
    assert_allclose(w, [2.0, 5.0])
    assert_allclose(v @ jnp.transpose(v), jnp.eye(2), atol=1e-5)
    w_only = mod.eigh(a, eigvals_only=True)
    assert_allclose(w_only, [2.0, 5.0])


def test_eigh_tridiagonal() -> None:
    """Test eigh_tridiagonal."""
    d = jnp.array([2.0, 3.0])
    e = jnp.array([0.0])
    w = mod.eigh_tridiagonal(d, e, eigvals_only=True)
    assert_allclose(w, [2.0, 3.0])
    w2, v2 = mod.eigh_tridiagonal(d, e, eigvals_only=False)
    assert_allclose(w2, [2.0, 3.0])
    assert v2.shape == (2, 2)


def test_expm() -> None:
    """Test expm."""
    a = jnp.zeros((2, 2))
    assert_allclose(mod.expm(a), jnp.eye(2))


def test_expm_frechet() -> None:
    """Test expm_frechet."""
    a = jnp.zeros((2, 2))
    e = jnp.eye(2)
    ea, frechet = mod.expm_frechet(a, e, compute_expm=True)
    assert_allclose(ea, jnp.eye(2))
    assert_allclose(frechet, jnp.eye(2))
    only_f = mod.expm_frechet(a, e, compute_expm=False)
    assert_allclose(only_f, jnp.eye(2))


def test_funm() -> None:
    """Test funm."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    res = mod.funm(a, jnp.sqrt)
    assert_allclose(res, [[2.0, 0.0], [0.0, 3.0]])
    res_disp, err = mod.funm(a, jnp.sqrt, disp=False)
    assert_allclose(res_disp, [[2.0, 0.0], [0.0, 3.0]])
    assert err == 0.0


def test_hessenberg() -> None:
    """Test hessenberg."""
    a = jnp.array([[2.0, 1.0], [0.0, 3.0]])
    h = mod.hessenberg(a)
    assert_allclose(h, a)
    h_q, q_h = mod.hessenberg(a, calc_q=True)
    assert_allclose(h_q, a)
    assert_allclose(q_h, jnp.eye(2))


def test_hilbert() -> None:
    """Test hilbert."""
    h = mod.hilbert(2)
    assert_allclose(h, [[1.0, 0.5], [0.5, 1.0 / 3.0]])


def test_inv() -> None:
    """Test inv."""
    a = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    assert_allclose(mod.inv(a), [[0.5, 0.0], [0.0, 0.25]])


def test_lu() -> None:
    """Test lu."""
    a = jnp.eye(2)
    p, l_m, u_m = mod.lu(a)
    assert_allclose(p @ l_m @ u_m, a)
    pl, u2 = mod.lu(a, permute_l=True)
    assert_allclose(pl @ u2, a)


def test_lu_factor() -> None:
    """Test lu_factor."""
    a = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    lu_mat, piv = mod.lu_factor(a)
    assert lu_mat.shape == (2, 2)
    assert piv.shape == (2,)


def test_lu_solve() -> None:
    """Test lu_solve."""
    a = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    b = jnp.array([6.0, 8.0])
    luf = mod.lu_factor(a)
    x = mod.lu_solve(luf, b)
    assert_allclose(x, [3.0, 2.0])
    x_t = mod.lu_solve(luf, b, trans=1)
    assert_allclose(x_t, [3.0, 2.0])


def test_pascal() -> None:
    """Test pascal."""
    assert_allclose(mod.pascal(3, kind="lower"), [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
    assert_allclose(mod.pascal(3, kind="upper"), [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    assert_allclose(mod.pascal(3, kind="symmetric"), [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 6.0]])


def test_polar() -> None:
    """Test polar."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    u, p = mod.polar(a)
    assert_allclose(u @ p, a)


def test_qr() -> None:
    """Test qr."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    q, r = mod.qr(a)
    assert_allclose(q @ r, a)


def test_rsf2csf() -> None:
    """Test rsf2csf."""
    t_mat = jnp.eye(2)
    z_mat = jnp.eye(2)
    t_out, z_out = mod.rsf2csf(t_mat, z_mat)
    assert_allclose(t_out, t_mat)
    assert_allclose(z_out, z_mat)


def test_schur() -> None:
    """Test schur."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    t_mat, z_mat = mod.schur(a)
    assert_allclose(z_mat @ t_mat @ jnp.transpose(z_mat), a)


def test_solve() -> None:
    """Test solve."""
    a = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    b = jnp.array([6.0, 8.0])
    assert_allclose(mod.solve(a, b), [3.0, 2.0])


def test_solve_sylvester() -> None:
    """Test solve_sylvester."""
    a = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    b = jnp.array([[3.0, 0.0], [0.0, 4.0]])
    q = jnp.array([[4.0, 5.0], [10.0, 12.0]])
    x = mod.solve_sylvester(a, b, q)
    assert_allclose(a @ x + x @ b, q, atol=1e-5)


def test_solve_triangular() -> None:
    """Test solve_triangular."""
    a = jnp.array([[2.0, 0.0], [1.0, 4.0]])
    b = jnp.array([4.0, 6.0])
    assert_allclose(mod.solve_triangular(a, b, lower=True), [2.0, 1.0])
    assert_allclose(mod.solve_triangular(jnp.transpose(a), b, trans="T"), [2.0, 1.0])


def test_sqrtm() -> None:
    """Test sqrtm."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    assert_allclose(mod.sqrtm(a), [[2.0, 0.0], [0.0, 3.0]])


def test_svd() -> None:
    """Test svd."""
    a = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    u, s, vh = mod.svd(a)
    assert_allclose(u @ jnp.diag(s) @ vh, a)


def test_toeplitz() -> None:
    """Test toeplitz."""
    c = jnp.array([1.0, 2.0, 3.0])
    r = jnp.array([1.0, 4.0])
    assert_allclose(mod.toeplitz(c, r), [[1.0, 4.0], [2.0, 1.0], [3.0, 2.0]])
    assert_allclose(mod.toeplitz(jnp.array([1.0, 2.0])), [[1.0, 2.0], [2.0, 1.0]])


def test_cholesky_solve_and_lu_helpers() -> None:
    """Test cholesky_solve, lu_reconstruct, and lu_matrix_inverse."""
    a = jnp.array([[4.0, 0.0], [0.0, 9.0]])
    c, lower = mod.cho_factor(a, True)
    b = jnp.array([8.0, 18.0])
    x = mod.cholesky_solve(c, b, lower)
    assert_allclose(x, [2.0, 2.0])

    eye2 = jnp.eye(2)
    rec = mod.lu_reconstruct(eye2, eye2, a)
    assert_allclose(rec, a)
    rec_tup = mod.lu_reconstruct((eye2, eye2, a))
    assert_allclose(rec_tup, a)

    inv_a = mod.lu_matrix_inverse(a)
    assert_allclose(inv_a, [[0.25, 0.0], [0.0, 1.0 / 9.0]])

    with pytest.raises(Exception):
        mod.lu_reconstruct(eye2, jnp.array([0, 1]))
    with pytest.raises(Exception):
        mod.lu_matrix_inverse(eye2, jnp.array([0, 1]))
