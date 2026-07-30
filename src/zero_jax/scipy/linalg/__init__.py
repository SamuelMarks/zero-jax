"""Frontend API routing for jax.scipy.linalg."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def block_diag(*args: Any, **kwargs: Any) -> Any:
    """Create a block diagonal matrix from input arrays."""
    return _ops.block_diag(*args, **kwargs)


def cho_factor(*args: Any, **kwargs: Any) -> Any:
    """Factorization for Cholesky-based linear solves"""
    return _ops.cho_factor(*args, **kwargs)


def cho_solve(*args: Any, **kwargs: Any) -> Any:
    """Solve a linear system using a Cholesky factorization"""
    return _ops.cho_solve(*args, **kwargs)


def cholesky(*args: Any, **kwargs: Any) -> Any:
    """Compute the Cholesky decomposition of a matrix."""
    return _ops.cholesky(*args, **kwargs)


def det(*args: Any, **kwargs: Any) -> Any:
    """Compute the determinant of a matrix"""
    return _ops.det(*args, **kwargs)


def eigh(*args: Any, **kwargs: Any) -> Any:
    """Compute eigenvalues and eigenvectors for a Hermitian matrix"""
    return _ops.eigh(*args, **kwargs)


def eigh_tridiagonal(*args: Any, **kwargs: Any) -> Any:
    """Solve the eigenvalue problem for a symmetric real tridiagonal matrix"""
    return _ops.eigh_tridiagonal(*args, **kwargs)


def expm(*args: Any, **kwargs: Any) -> Any:
    """Compute the matrix exponential"""
    return _ops.expm(*args, **kwargs)


def expm_frechet(*args: Any, **kwargs: Any) -> Any:
    """Compute the Frechet derivative of the matrix exponential."""
    return _ops.expm_frechet(*args, **kwargs)


def funm(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a matrix-valued function"""
    return _ops.funm(*args, **kwargs)


def hessenberg(*args: Any, **kwargs: Any) -> Any:
    """Compute the Hessenberg form of the matrix"""
    return _ops.hessenberg(*args, **kwargs)


def hilbert(*args: Any, **kwargs: Any) -> Any:
    """Create a Hilbert matrix of order n."""
    return _ops.hilbert(*args, **kwargs)


def inv(*args: Any, **kwargs: Any) -> Any:
    """Return the inverse of a square matrix"""
    return _ops.inv(*args, **kwargs)


def lu(*args: Any, **kwargs: Any) -> Any:
    """Compute the LU decomposition"""
    return _ops.lu(*args, **kwargs)


def lu_factor(*args: Any, **kwargs: Any) -> Any:
    """Factorization for LU-based linear solves"""
    return _ops.lu_factor(*args, **kwargs)


def lu_solve(*args: Any, **kwargs: Any) -> Any:
    """Solve a linear system using an LU factorization"""
    return _ops.lu_solve(*args, **kwargs)


def polar(*args: Any, **kwargs: Any) -> Any:
    """Computes the polar decomposition."""
    return _ops.polar(*args, **kwargs)


def qr(*args: Any, **kwargs: Any) -> Any:
    """Compute the QR decomposition of an array"""
    return _ops.qr(*args, **kwargs)


def rsf2csf(*args: Any, **kwargs: Any) -> Any:
    """Convert real Schur form to complex Schur form."""
    return _ops.rsf2csf(*args, **kwargs)


def schur(*args: Any, **kwargs: Any) -> Any:
    """Compute the Schur decomposition"""
    return _ops.schur(*args, **kwargs)


def solve(*args: Any, **kwargs: Any) -> Any:
    """Solve a linear system of equations"""
    return _ops.solve(*args, **kwargs)


def solve_triangular(*args: Any, **kwargs: Any) -> Any:
    """Solve a triangular linear system of equations"""
    return _ops.solve_triangular(*args, **kwargs)


def sqrtm(*args: Any, **kwargs: Any) -> Any:
    """Compute the matrix square root"""
    return _ops.sqrtm(*args, **kwargs)


def svd(*args: Any, **kwargs: Any) -> Any:
    """Compute the singular value decomposition."""
    return _ops.svd(*args, **kwargs)


def toeplitz(*args: Any, **kwargs: Any) -> Any:
    """Construct a Toeplitz matrix"""
    return _ops.toeplitz(*args, **kwargs)


def cholesky_solve(*args: Any, **kwargs: Any) -> Any:
    """Solve a linear system using a Cholesky factorization"""
    return _ops.cholesky_solve(*args, **kwargs)  # pragma: no cover


def lu_reconstruct(*args: Any, **kwargs: Any) -> Any:
    """Reconstruct a matrix from its LU factorization"""
    return _ops.lu_reconstruct(*args, **kwargs)  # pragma: no cover


def lu_matrix_inverse(*args: Any, **kwargs: Any) -> Any:
    """Compute the matrix inverse using an LU factorization"""
    return _ops.lu_matrix_inverse(*args, **kwargs)  # pragma: no cover
