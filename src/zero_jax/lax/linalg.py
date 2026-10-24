"""Module documentation."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax.mock_p import MockPrimitive
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cholesky(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cholesky.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cholesky(_to_tensor(a)))


cholesky_p = MockPrimitive("cholesky")


def eig(x: ArrayLikeSpec, compute_v: bool = True) -> ArrayLikeSpec:
    """JAX API implementation for eig.

    Args:
        x: Argument x.
        compute_v: Argument compute_v.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.eig(_to_tensor(x), compute_v=compute_v)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


eig_p = MockPrimitive("eig")


def eigh(
    x: ArrayLikeSpec,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
) -> ArrayLikeSpec:
    """JAX API implementation for eigh.

    Args:
        x: Argument x.
        lower: Argument lower.
        symmetrize_input: Argument symmetrize_input.
        sort_eigenvalues: Argument sort_eigenvalues.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.eigh(
        _to_tensor(x),
        lower=lower,
        symmetrize_input=symmetrize_input,
        sort_eigenvalues=sort_eigenvalues,
    )
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


eigh_p = MockPrimitive("eigh")


def hessenberg(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for hessenberg.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.hessenberg(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


hessenberg_p = MockPrimitive("hessenberg")


def householder_product(a: ArrayLikeSpec, taus: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for householder_product.

    Args:
        a: Argument a.
        taus: Argument taus.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.householder_product(_to_tensor(a), _to_tensor(taus)))


householder_product_p = MockPrimitive("householder_product")


def lu(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for lu.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.lu(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


lu_p = MockPrimitive("lu")


def lu_pivots_to_permutation(pivots: ArrayLikeSpec, permutation_size: int) -> ArrayLikeSpec:
    """JAX API implementation for lu_pivots_to_permutation.

    Args:
        pivots: Argument pivots.
        permutation_size: Argument permutation_size.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.lu_pivots_to_permutation(_to_tensor(pivots), permutation_size=permutation_size))


def qdwh(x: ArrayLikeSpec, is_hermitian: bool = False, max_iterations: int = 100) -> ArrayLikeSpec:
    """JAX API implementation for qdwh.

    Args:
        x: Argument x.
        is_hermitian: Argument is_hermitian.
        max_iterations: Argument max_iterations.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.qdwh(_to_tensor(x), is_hermitian=is_hermitian, max_iterations=max_iterations)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


def qr(a: ArrayLikeSpec, full_matrices: bool = True) -> ArrayLikeSpec:
    """JAX API implementation for qr.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.qr(_to_tensor(a), full_matrices=full_matrices)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


qr_p = MockPrimitive("qr")


def schur(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for schur.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.schur(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


schur_p = MockPrimitive("schur")


def svd(a: ArrayLikeSpec, full_matrices: bool = True, compute_uv: bool = True) -> ArrayLikeSpec:
    """JAX API implementation for svd.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.
        compute_uv: Argument compute_uv.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.svd(_to_tensor(a), full_matrices=full_matrices, compute_uv=compute_uv)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


svd_p = MockPrimitive("svd")


def triangular_solve(
    a: ArrayLikeSpec,
    b: ArrayLikeSpec,
    left_side: bool = False,
    lower: bool = False,
    transpose_a: bool = False,
    conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> ArrayLikeSpec:
    """JAX API implementation for triangular_solve.

    Args:
        a: Argument a.
        b: Argument b.
        left_side: Argument left_side.
        lower: Argument lower.
        transpose_a: Argument transpose_a.
        conjugate_a: Argument conjugate_a.
        unit_diagonal: Argument unit_diagonal.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.triangular_solve(
            _to_tensor(a),
            _to_tensor(b),
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=conjugate_a,
            unit_diagonal=unit_diagonal,
        )
    )


triangular_solve_p = MockPrimitive("triangular_solve")


def tridiagonal(a: ArrayLikeSpec, lower: bool = True) -> ArrayLikeSpec:
    """JAX API implementation for tridiagonal.

    Args:
        a: Argument a.
        lower: Argument lower.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.tridiagonal(_to_tensor(a), lower=lower)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


tridiagonal_p = MockPrimitive("tridiagonal")


def tridiagonal_solve(dl: ArrayLikeSpec, d: ArrayLikeSpec, du: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for tridiagonal_solve.

    Args:
        dl: Argument dl.
        d: Argument d.
        du: Argument du.
        b: Argument b.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tridiagonal_solve(_to_tensor(dl), _to_tensor(d), _to_tensor(du), _to_tensor(b)))


tridiagonal_solve_p = MockPrimitive("tridiagonal_solve")


__all__ = [
    "cholesky",
    "cholesky_p",
    "eig",
    "eig_p",
    "eigh",
    "eigh_p",
    "hessenberg",
    "hessenberg_p",
    "householder_product",
    "householder_product_p",
    "lu",
    "lu_p",
    "lu_pivots_to_permutation",
    "qdwh",
    "qr",
    "qr_p",
    "schur",
    "schur_p",
    "svd",
    "svd_p",
    "triangular_solve",
    "triangular_solve_p",
    "tridiagonal",
    "tridiagonal_p",
    "tridiagonal_solve",
    "tridiagonal_solve_p",
]
