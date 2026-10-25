"""Frontend API routing for jax.scipy.linalg."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
    arange,
    asarray,
    float32,
    ones_like,
    zeros,
)


def _call_linalg(proxy_fn: ConfigValue, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Invoke a linear algebra proxy operation converting inputs to ``Tensor`` and wrapping outputs.

    Args:
        proxy_fn: Proxy callable.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Wrapped ``ndarray`` or tuple of ``ndarray``s.
    """
    t_args = [_to_tensor(a) for a in args]
    res = proxy_fn(*t_args, **kwargs)
    return _wrap(res)


def block_diag(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a block diagonal matrix from input arrays.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.block_diag, *args, **kwargs)


def cho_factor(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Factorization for Cholesky-based linear solves.

    Args:
        *args: Input parameter ``args`` (``a``, ``lower=False``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        from zero_jax.numpy.linalg import cholesky as _jnp_cholesky

        lower = bool(args[1]) if len(args) > 1 else bool(kwargs.get("lower", False))
        c = _jnp_cholesky(args[0], upper=not lower)
        return c, lower
    return _ops.cho_factor(*args, **kwargs)


def cho_solve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a linear system using a Cholesky factorization.

    Args:
        *args: Input parameter ``args`` (``(c, lower)``, ``b``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and isinstance(args[0], tuple) and len(args[0]) == 2:
        c, lower = args[0]
        b = args[1]
        from zero_jax.numpy.linalg import matrix_transpose
        from zero_jax.numpy.linalg import solve as _jnp_solve

        a = c @ matrix_transpose(c) if lower else matrix_transpose(c) @ c
        return _jnp_solve(a, b)
    return _call_linalg(_ops.cho_solve, *args, **kwargs)


def cholesky(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Cholesky decomposition of a matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.cholesky, *args, **kwargs)


def det(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the determinant of a matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.det, *args, **kwargs)


def eigh(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute eigenvalues and eigenvectors for a Hermitian matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.eigh, *args, **kwargs)


def eigh_tridiagonal(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve the eigenvalue problem for a symmetric real tridiagonal matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.eigh_tridiagonal, *args, **kwargs)


def expm(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the matrix exponential.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.expm, *args, **kwargs)


def expm_frechet(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Frechet derivative of the matrix exponential.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and not kwargs:
        a = asarray(args[0], dtype=float32)
        e = asarray(args[1], dtype=float32)
        ea = expm(a)
        return ea, 0.5 * (ea @ e + e @ ea)
    return _call_linalg(_ops.expm_frechet, *args, **kwargs)


def funm(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Evaluate a matrix-valued function.

    Args:
        *args: Input parameter ``args`` (``A``, ``func``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and callable(args[1]):
        from zero_jax.numpy.linalg import eigh as _jnp_eigh
        from zero_jax.numpy.linalg import matrix_transpose

        a = asarray(args[0], dtype=float32)
        w, v = _jnp_eigh(a)
        fw = args[1](w)
        return (v * fw) @ matrix_transpose(v)
    return _ops.funm(*args, **kwargs)


def hessenberg(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Hessenberg form of the matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.hessenberg, *args, **kwargs)


def hilbert(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a Hilbert matrix of order n.

    Args:
        *args: Input parameter ``args`` (``n``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and isinstance(args[0], int) and not kwargs:
        n = int(args[0])
        idx = arange(1, n + 1, dtype=float32)
        return 1.0 / (idx[:, None] + idx[None, :] - 1.0)
    return _ops.hilbert(*args, **kwargs)


def inv(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the inverse of a square matrix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.inv, *args, **kwargs)


def lu(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the LU decomposition.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.lu, *args, **kwargs)


def lu_factor(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Factorization for LU-based linear solves.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.lu_factor, *args, **kwargs)


def lu_solve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a linear system using an LU factorization.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and isinstance(args[0], tuple) and len(args[0]) == 2:
        lu_mat, piv = args[0]
        return _wrap(_ops.lu_solve(_to_tensor(lu_mat), _to_tensor(piv), _to_tensor(args[1]), **kwargs))
    return _call_linalg(_ops.lu_solve, *args, **kwargs)


def polar(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the polar decomposition.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.polar, *args, **kwargs)


def qr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the QR decomposition of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.qr, *args, **kwargs)


def rsf2csf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert real Schur form to complex Schur form.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and not kwargs:
        return asarray(args[0]), asarray(args[1])
    return _call_linalg(_ops.rsf2csf, *args, **kwargs)


def schur(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the Schur decomposition.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.schur, *args, **kwargs)


def solve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a linear system of equations.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.solve, *args, **kwargs)


def solve_triangular(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a triangular linear system of equations.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        from zero_jax.numpy.linalg import matrix_transpose
        from zero_jax.numpy.linalg import solve as _jnp_solve

        a = asarray(args[0], dtype=float32)
        b = asarray(args[1], dtype=float32)
        trans = kwargs.get("trans", 0)
        if trans in (1, 2, "T", "C"):
            a = matrix_transpose(a)
        return _jnp_solve(a, b)
    return _call_linalg(_ops.solve_triangular, *args, **kwargs)


def sqrtm(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the matrix square root.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.sqrtm, *args, **kwargs)


def svd(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the singular value decomposition.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.svd, *args, **kwargs)


def toeplitz(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct a Toeplitz matrix.

    Args:
        *args: Input parameter ``args`` (``c``, ``r=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and not kwargs:
        c = asarray(args[0], dtype=float32)
        r = asarray(args[1], dtype=float32) if len(args) > 1 and args[1] is not None else c
        n = c.shape[0]
        m = r.shape[0]
        out = zeros((n, m), dtype=c.dtype)
        for i in range(n):
            for j in range(m):
                out[i, j] = c[i - j] if i >= j else r[j - i]
        return out
    return _call_linalg(_ops.toeplitz, *args, **kwargs)


def cholesky_solve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Solve a linear system using a Cholesky factorization.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.cholesky_solve, *args, **kwargs)


def lu_reconstruct(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reconstruct a matrix from its LU factorization.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.lu_reconstruct, *args, **kwargs)


def lu_matrix_inverse(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the matrix inverse using an LU factorization.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_linalg(_ops.lu_matrix_inverse, *args, **kwargs)
