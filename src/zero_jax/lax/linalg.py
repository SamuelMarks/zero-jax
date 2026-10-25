"""Linear algebra primitives for ``zero_jax.lax.linalg``."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax.mock_p import MockPrimitive
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cholesky(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Args:
        a: Input symmetric/Hermitian positive-definite matrix.

    Returns:
        ArrayLikeSpec: Lower-triangular Cholesky factor of ``a``.
    """
    return _wrap(ops.cholesky(_to_tensor(a)))


cholesky_p = MockPrimitive("cholesky")


def eig(x: ArrayLikeSpec, compute_v: bool = True) -> ArrayLikeSpec:
    """Compute the eigendecomposition of a general square matrix.

    Args:
        x: Input square matrix or batch of square matrices.
        compute_v: Whether to compute right/left eigenvectors in addition to eigenvalues.

    Returns:
        ArrayLikeSpec: Eigenvalues and (optionally) eigenvectors of ``x``.
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
    """Compute the eigendecomposition of a symmetric or Hermitian matrix.

    Args:
        x: Input symmetric or Hermitian matrix.
        lower: Whether to use the lower triangle of ``x``.
        symmetrize_input: Whether to symmetrize ``x`` before decomposition.
        sort_eigenvalues: Whether to sort eigenvalues in ascending order.

    Returns:
        ArrayLikeSpec: Tuple of ``(eigenvectors, eigenvalues)`` or ``(eigenvalues, eigenvectors)``.
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
    """Reduce a square matrix to upper Hessenberg form via orthogonal similarity transformations.

    Args:
        a: Input square matrix.

    Returns:
        ArrayLikeSpec: Tuple of Hessenberg matrix and elementary reflector factors ``taus``.
    """
    res = ops.hessenberg(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


hessenberg_p = MockPrimitive("hessenberg")


def householder_product(a: ArrayLikeSpec, taus: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the product of elementary Householder reflectors.

    Args:
        a: Matrix whose lower triangle encodes the Householder reflector vectors.
        taus: Scalar scale factors for the elementary reflectors.

    Returns:
        ArrayLikeSpec: Orthogonal/unitary matrix ``Q``.
    """
    return _wrap(ops.householder_product(_to_tensor(a), _to_tensor(taus)))


householder_product_p = MockPrimitive("householder_product")


def lu(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the LU decomposition with partial pivoting.

    Args:
        a: Input matrix or batch of matrices.

    Returns:
        ArrayLikeSpec: Tuple of ``(lu, pivots, permutation)``.
    """
    res = ops.lu(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


lu_p = MockPrimitive("lu")


def lu_pivots_to_permutation(pivots: ArrayLikeSpec, permutation_size: int) -> ArrayLikeSpec:
    """Convert pivot indices from LU factorization into a permutation index vector.

    Args:
        pivots: 1D or batched array of swap pivot indices.
        permutation_size: Length of the output permutation vector.

    Returns:
        ArrayLikeSpec: Permutation index array.
    """
    return _wrap(ops.lu_pivots_to_permutation(_to_tensor(pivots), permutation_size=permutation_size))


def qdwh(x: ArrayLikeSpec, is_hermitian: bool = False, max_iterations: int = 100) -> ArrayLikeSpec:
    """Compute the polar decomposition using QR-based dynamically weighted Halley (QDWH) iteration.

    Args:
        x: Input matrix to decompose.
        is_hermitian: Whether ``x`` is known to be Hermitian.
        max_iterations: Maximum number of Halley iterations.

    Returns:
        ArrayLikeSpec: Polar decomposition factors ``(u, h, num_iters, is_converged)``.
    """
    res = ops.qdwh(_to_tensor(x), is_hermitian=is_hermitian, max_iterations=max_iterations)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


def qr(a: ArrayLikeSpec, full_matrices: bool = True) -> ArrayLikeSpec:
    """Compute the QR decomposition of a matrix.

    Args:
        a: Input matrix or batch of matrices.
        full_matrices: Whether to compute full-sized ``Q`` and ``R`` matrices.

    Returns:
        ArrayLikeSpec: Tuple of ``(Q, R)``.
    """
    res = ops.qr(_to_tensor(a), full_matrices=full_matrices)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


qr_p = MockPrimitive("qr")


def schur(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Schur decomposition of a square matrix.

    Args:
        a: Input square matrix.

    Returns:
        ArrayLikeSpec: Tuple of Schur form ``T`` and unitary Schur vectors ``Q``.
    """
    res = ops.schur(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


schur_p = MockPrimitive("schur")


def svd(a: ArrayLikeSpec, full_matrices: bool = True, compute_uv: bool = True) -> ArrayLikeSpec:
    """Compute the Singular Value Decomposition of a matrix.

    Args:
        a: Input matrix or batch of matrices.
        full_matrices: Whether to compute full or reduced singular vector matrices.
        compute_uv: Whether to compute left and right singular vectors in addition to singular values.

    Returns:
        ArrayLikeSpec: Singular values ``s`` or tuple ``(s, u, vt)``.
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
    """Solve a triangular linear system ``op(a) @ x = b`` or ``x @ op(a) = b``.

    Args:
        a: Triangular coefficient matrix.
        b: Right-hand side matrix.
        left_side: If ``True``, solve ``op(a) @ x = b``; otherwise ``x @ op(a) = b``.
        lower: Whether ``a`` is lower-triangular.
        transpose_a: Whether to transpose ``a`` before solving.
        conjugate_a: Whether to complex-conjugate ``a`` before solving.
        unit_diagonal: Whether to assume the diagonal of ``a`` is all ones.

    Returns:
        ArrayLikeSpec: Solution array ``x``.
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
    """Reduce a symmetric/Hermitian matrix to tridiagonal form.

    Args:
        a: Input symmetric or Hermitian matrix.
        lower: Whether to read from the lower triangle of ``a``.

    Returns:
        ArrayLikeSpec: Tuple of ``(a_out, d, e, taus)``.
    """
    res = ops.tridiagonal(_to_tensor(a), lower=lower)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)


tridiagonal_p = MockPrimitive("tridiagonal")


def tridiagonal_solve(dl: ArrayLikeSpec, d: ArrayLikeSpec, du: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a tridiagonal linear system given lower, main, and upper diagonals.

    Args:
        dl: Sub-diagonal vector (with first element 0).
        d: Main diagonal vector.
        du: Super-diagonal vector (with last element 0).
        b: Right-hand side vector or matrix.

    Returns:
        ArrayLikeSpec: Solution to the tridiagonal system.
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
