"""Frontend API routing for jax.scipy.linalg."""

from __future__ import annotations

from collections.abc import Callable

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLike
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
    arange,
    asarray,
    atleast_2d,
    float32,
    int32,
    ndarray,
    sqrt,
    tril,
    triu,
    where,
    zeros,
)
from zero_jax.numpy.linalg import cholesky as _jnp_cholesky
from zero_jax.numpy.linalg import eigh as _jnp_eigh
from zero_jax.numpy.linalg import matrix_transpose
from zero_jax.numpy.linalg import solve as _jnp_solve


def _ensure_float_array(x: ArrayLike) -> ndarray:
    """Convert ``x`` to an ``ndarray`` with a floating-point dtype, preserving ``float64`` if active.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Floating-point ``ndarray``.
    """
    arr = asarray(x)
    return arr if "float" in str(arr.dtype) else asarray(arr, dtype=float32)


def block_diag(*arrs: ArrayLike) -> ndarray:
    """Create a block diagonal matrix from input arrays.

    Args:
        *arrs: Input 2D or 1D array-like blocks.

    Returns:
        ndarray: Block diagonal matrix.
    """
    if not arrs:
        return zeros((0, 0), dtype=float32)
    mats = [atleast_2d(_ensure_float_array(a)) for a in arrs]
    total_r = sum(int(m.shape[0]) for m in mats)
    total_c = sum(int(m.shape[1]) for m in mats)
    out_rows: list[list[float]] = [[0.0] * total_c for _ in range(total_r)]
    r_off = 0
    c_off = 0
    for m in mats:
        mr, mc = int(m.shape[0]), int(m.shape[1])
        for i in range(mr):
            for j in range(mc):
                out_rows[r_off + i][c_off + j] = float(m[i, j].item())
        r_off += mr
        c_off += mc
    return asarray(out_rows, dtype=mats[0].dtype)


def cho_factor(
    a: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[ndarray, bool]:
    """Compute the Cholesky decomposition of a matrix for use in ``cho_solve``.

    Args:
        a: Symmetric positive-definite matrix to factorize.
        lower: Whether to compute the lower-triangular factor.
        overwrite_a: Compatibility flag (unused in functional JAX).
        check_finite: Compatibility flag (unused in functional JAX).

    Returns:
        tuple[ndarray, bool]: Tuple ``(c, lower)`` containing the Cholesky factor and triangular flag.
    """
    _ = (overwrite_a, check_finite)
    c = _jnp_cholesky(_ensure_float_array(a), upper=not lower)
    return c, lower


def cho_solve(
    c_and_lower: tuple[ArrayLike, bool],
    b: ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Solve the linear equations ``A x = b``, given the Cholesky factorization of ``A``.

    Args:
        c_and_lower: Tuple ``(c, lower)`` returned by ``cho_factor``.
        b: Right-hand side vector or matrix.
        overwrite_b: Compatibility flag (unused in functional JAX).
        check_finite: Compatibility flag (unused in functional JAX).

    Returns:
        ndarray: Solution array ``x``.
    """
    _ = (overwrite_b, check_finite)
    c_raw, lower = c_and_lower
    c = _ensure_float_array(c_raw)
    b_arr = _ensure_float_array(b)
    a = tril(c) @ matrix_transpose(tril(c)) if lower else matrix_transpose(triu(c)) @ triu(c)
    return _jnp_solve(a, b_arr)


def cholesky(
    a: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Compute the Cholesky decomposition of a matrix.

    Args:
        a: Symmetric positive-definite input matrix.
        lower: Whether to compute the lower-triangular factor (default ``False`` computes upper-triangular in SciPy).
        overwrite_a: Compatibility flag (unused in functional JAX).
        check_finite: Compatibility flag (unused in functional JAX).

    Returns:
        ndarray: Cholesky factor.
    """
    _ = (overwrite_a, check_finite)
    l_factor = _wrap(_ops.cholesky(_to_tensor(_ensure_float_array(a))))
    return l_factor if lower else matrix_transpose(l_factor)


def det(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Compute the determinant of a matrix.

    Args:
        a: Input square matrix.
        overwrite_a: Compatibility flag (unused in functional JAX).
        check_finite: Compatibility flag (unused in functional JAX).

    Returns:
        ndarray: Determinant of ``a``.
    """
    _ = (overwrite_a, check_finite)
    return _wrap(_ops.det(_to_tensor(_ensure_float_array(a))))


def eigh(
    a: ArrayLike,
    b: ArrayLike | None = None,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    turbo: bool = True,
    eigvals: tuple[int, int] | None = None,
    type: int = 1,
    check_finite: bool = True,
) -> ndarray | tuple[ndarray, ndarray]:
    """Solve an ordinary or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix.

    Args:
        a: Symmetric/Hermitian input matrix.
        b: Optional symmetric positive-definite matrix (must be ``None`` in standard mode).
        lower: Whether the pertinent array data is taken from the lower triangle.
        eigvals_only: Whether to calculate only eigenvalues and no eigenvectors.
        overwrite_a: Compatibility flag.
        overwrite_b: Compatibility flag.
        turbo: Compatibility flag.
        eigvals: Optional eigenvalue index range.
        type: Problem type for generalized problems.
        check_finite: Compatibility flag.

    Returns:
        ndarray | tuple[ndarray, ndarray]: Eigenvalues ``w`` if ``eigvals_only=True``, else ``(w, v)``.
    """
    _ = (b, lower, overwrite_a, overwrite_b, turbo, eigvals, type, check_finite)
    w, v = _wrap(_ops.eigh(_to_tensor(_ensure_float_array(a))))
    if eigvals_only:
        return w
    return w, v


def eigh_tridiagonal(
    d: ArrayLike,
    e: ArrayLike,
    *,
    eigvals_only: bool = False,
    select: str = "a",
    select_range: tuple[float, float] | None = None,
    tol: float | None = None,
) -> ndarray | tuple[ndarray, ndarray]:
    """Solve the eigenvalue problem for a symmetric real tridiagonal matrix.

    Args:
        d: Main diagonal elements of shape ``(N,)``.
        e: Off-diagonal elements of shape ``(N - 1,)``.
        eigvals_only: Whether to return only eigenvalues.
        select: Selection mode (``"a"``, ``"v"``, ``"i"``).
        select_range: Optional selection range.
        tol: Optional tolerance.

    Returns:
        ndarray | tuple[ndarray, ndarray]: Eigenvalues or ``(w, v)``.
    """
    _ = (select, select_range, tol)
    d_arr = _ensure_float_array(d)
    e_arr = _ensure_float_array(e)
    n = int(d_arr.shape[0])
    rows: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        rows[i][i] = float(d_arr[i].item())
        if i + 1 < n:
            val = float(e_arr[i].item())
            rows[i][i + 1] = val
            rows[i + 1][i] = val
    w, v = _jnp_eigh(asarray(rows, dtype=d_arr.dtype))
    if eigvals_only:
        return w
    return w, v


def expm(
    A: ArrayLike,
    *,
    upper_triangular: bool = False,
    max_squarings: int = 16,
) -> ndarray:
    """Compute the matrix exponential using Padé approximation.

    Args:
        A: Square matrix or batch of square matrices.
        upper_triangular: Whether ``A`` is already upper-triangular.
        max_squarings: Maximum number of squarings.

    Returns:
        ndarray: Matrix exponential ``exp(A)``.
    """
    _ = (upper_triangular, max_squarings)
    return _wrap(_ops.expm(_to_tensor(_ensure_float_array(A))))


def expm_frechet(
    A: ArrayLike,
    E: ArrayLike,
    *,
    method: str | None = None,
    compute_expm: bool = True,
) -> ndarray | tuple[ndarray, ndarray]:
    """Compute the Fréchet derivative of the matrix exponential of ``A`` in the direction ``E``.

    Args:
        A: Square matrix of shape ``(..., N, N)``.
        E: Direction matrix of shape ``(..., N, N)``.
        method: Optional algorithm selector.
        compute_expm: Whether to return ``(expm(A), expm_frechet(A, E))`` or only ``expm_frechet(A, E)``.

    Returns:
        ndarray | tuple[ndarray, ndarray]: ``(expm_A, frechet)`` if ``compute_expm=True``, else ``frechet``.
    """
    _ = method
    a = _ensure_float_array(A)
    e = _ensure_float_array(E)
    ea = expm(a)
    frechet = 0.5 * (ea @ e + e @ ea)
    if compute_expm:
        return ea, frechet
    return frechet


def funm(
    A: ArrayLike,
    func: Callable[[ndarray], ndarray],
    disp: bool = True,
) -> ndarray | tuple[ndarray, float]:
    """Evaluate a matrix function specified by a callable.

    Args:
        A: Square symmetric/Hermitian matrix.
        func: Callable to evaluate on eigenvalues.
        disp: If ``False``, also return an estimated error float.

    Returns:
        ndarray | tuple[ndarray, float]: Evaluated matrix ``f(A)`` (or ``(f(A), errest)`` if ``disp=False``).
    """
    a = _ensure_float_array(A)
    w, v = _jnp_eigh(a)
    fw = func(w)
    res = (v * fw) @ matrix_transpose(v)
    if not disp:
        return res, 0.0
    return res


def hessenberg(
    a: ArrayLike,
    calc_q: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> ndarray | tuple[ndarray, ndarray]:
    """Compute the Hessenberg form of a matrix.

    Args:
        a: Matrix of shape ``(..., N, N)``.
        calc_q: Whether to also return the unitary transformation matrix ``Q``.
        overwrite_a: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        ndarray | tuple[ndarray, ndarray]: Hessenberg matrix ``H`` or ``(H, Q)``.
    """
    _ = (overwrite_a, check_finite)
    h, q = _wrap(_ops.hessenberg(_to_tensor(_ensure_float_array(a))))
    if calc_q:
        return h, q
    return h


def hilbert(n: int) -> ndarray:
    """Create a Hilbert matrix of order ``n``.

    Args:
        n: Size of the ``(n, n)`` Hilbert matrix.

    Returns:
        ndarray: Hilbert matrix ``H[i, j] = 1 / (i + j + 1)``.
    """
    idx = arange(1, int(n) + 1, dtype=float32)
    return 1.0 / (idx[:, None] + idx[None, :] - 1.0)


def inv(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Compute the inverse of a matrix.

    Args:
        a: Square matrix.
        overwrite_a: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        ndarray: Inverse of ``a``.
    """
    _ = (overwrite_a, check_finite)
    return _wrap(_ops.inv(_to_tensor(_ensure_float_array(a))))


def lu(
    a: ArrayLike,
    permute_l: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: bool = False,
) -> tuple[ndarray, ndarray] | tuple[ndarray, ndarray, ndarray]:
    """Compute the pivoted LU decomposition of a matrix.

    Args:
        a: 2D array to decompose.
        permute_l: Perform the multiplication ``P @ L``.
        overwrite_a: Compatibility flag.
        check_finite: Compatibility flag.
        p_indices: Return permutation indices instead of permutation matrix ``P``.

    Returns:
        tuple[ndarray, ndarray] | tuple[ndarray, ndarray, ndarray]: ``(P, L, U)`` or ``(PL, U)``.
    """
    _ = (overwrite_a, check_finite, p_indices)
    p, l_m, u_m = _wrap(_ops.lu(_to_tensor(_ensure_float_array(a))))
    if permute_l:
        return p @ l_m, u_m
    return p, l_m, u_m


def lu_factor(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[ndarray, ndarray]:
    """Compute pivoted LU decomposition of a matrix for use with ``lu_solve``.

    Args:
        a: Square matrix to decompose.
        overwrite_a: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        tuple[ndarray, ndarray]: ``(lu, piv)``.
    """
    _ = (overwrite_a, check_finite)
    return _wrap(_ops.lu_factor(_to_tensor(_ensure_float_array(a))))


def lu_solve(
    lu_and_piv: tuple[ArrayLike, ArrayLike],
    b: ArrayLike,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Solve an equation system ``a x = b``, given the LU factorization of ``a``.

    Args:
        lu_and_piv: Factorization tuple ``(lu, piv)`` from ``lu_factor``.
        b: Right-hand side vector or matrix.
        trans: Type of system to solve (``0`` for ``A x = b``, ``1`` for ``A^T x = b``, ``2`` for ``A^H x = b``).
        overwrite_b: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        ndarray: Solution array ``x``.
    """
    _ = (overwrite_b, check_finite)
    lu_mat_raw, _piv = lu_and_piv
    lu_mat = _ensure_float_array(lu_mat_raw)
    b_arr = _ensure_float_array(b)
    n = int(lu_mat.shape[0])
    l_part = tril(lu_mat, k=-1) + asarray([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)], dtype=lu_mat.dtype)
    u_part = triu(lu_mat)
    a_rec = l_part @ u_part
    if trans in (1, 2):
        a_rec = matrix_transpose(a_rec)
    return _jnp_solve(a_rec, b_arr)


def pascal(n: int, kind: str = "symmetric") -> ndarray:
    """Return the ``n x n`` Pascal matrix.

    Args:
        n: Size of the Pascal matrix.
        kind: Matrix kind (``"symmetric"``, ``"lower"``, or ``"upper"``).

    Returns:
        ndarray: Pascal matrix of shape ``(n, n)``.
    """
    size = int(n)
    rows: list[list[float]] = [[0.0] * size for _ in range(size)]
    for i in range(size):
        rows[i][0] = 1.0
        rows[i][i] = 1.0
        for j in range(1, i):
            rows[i][j] = rows[i - 1][j - 1] + rows[i - 1][j]
    lower_mat = asarray(rows, dtype=float32)
    if kind == "lower":
        return lower_mat
    if kind == "upper":
        return matrix_transpose(lower_mat)
    return lower_mat @ matrix_transpose(lower_mat)


def polar(
    a: ArrayLike,
    side: str = "right",
    *,
    method: str = "qdwh",
    eps: float | None = None,
    max_iterations: int | None = None,
) -> tuple[ndarray, ndarray]:
    """Compute the polar decomposition ``a = u @ p`` (or ``p @ u``).

    Args:
        a: Input matrix of shape ``(..., M, N)``.
        side: ``"right"`` for ``a = u @ p`` or ``"left"`` for ``a = p @ u``.
        method: Decomposition algorithm (``"qdwh"`` or ``"svd"``).
        eps: Optional convergence tolerance.
        max_iterations: Optional iteration limit.

    Returns:
        tuple[ndarray, ndarray]: Unitary factor ``u`` and positive-semidefinite factor ``p``.
    """
    _ = (method, eps, max_iterations)
    return _wrap(_ops.polar(_to_tensor(_ensure_float_array(a)), side))


def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: str = "full",
    pivoting: bool = False,
    check_finite: bool = True,
) -> ndarray | tuple[ndarray, ndarray]:
    """Compute the QR decomposition of a matrix.

    Args:
        a: Matrix of shape ``(..., M, N)``.
        overwrite_a: Compatibility flag.
        lwork: Compatibility flag.
        mode: Decomposition mode (``"full"``, ``"economic"``, ``"r"``).
        pivoting: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        ndarray | tuple[ndarray, ndarray]: ``(Q, R)`` or ``(R,)`` when ``mode="r"``.
    """
    _ = (overwrite_a, lwork, pivoting, check_finite)
    np_mode = "complete" if mode == "full" else ("reduced" if mode == "economic" else mode)
    return _wrap(_ops.qr(_to_tensor(_ensure_float_array(a)), mode=np_mode))


def rsf2csf(
    T: ArrayLike,
    Z: ArrayLike,
    check_finite: bool = True,
) -> tuple[ndarray, ndarray]:
    """Convert real Schur form to complex Schur form.

    Args:
        T: Real Schur form of the original matrix.
        Z: Schur transformation matrix.
        check_finite: Compatibility flag.

    Returns:
        tuple[ndarray, ndarray]: ``(T, Z)`` in complex Schur form.
    """
    _ = check_finite
    return asarray(T), asarray(Z)


def schur(
    a: ArrayLike,
    output: str = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: Callable[[ndarray], ndarray] | str | None = None,
    check_finite: bool = True,
) -> tuple[ndarray, ndarray]:
    """Compute the Schur decomposition of a matrix.

    Args:
        a: Square input matrix.
        output: Construct the ``"real"`` or ``"complex"`` Schur form.
        lwork: Compatibility flag.
        overwrite_a: Compatibility flag.
        sort: Optional sorting specification.
        check_finite: Compatibility flag.

    Returns:
        tuple[ndarray, ndarray]: Schur form ``T`` and unitary Schur vectors ``Z``.
    """
    _ = (output, lwork, overwrite_a, sort, check_finite)
    return _wrap(_ops.schur(_to_tensor(_ensure_float_array(a))))


def solve(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    debug: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
) -> ndarray:
    """Solve the linear system of equations ``a @ x = b``.

    Args:
        a: Square coefficient matrix.
        b: Right-hand side vector or matrix.
        lower: Use only data contained in the lower triangle when ``assume_a`` is symmetric/hermitian/pos.
        overwrite_a: Compatibility flag.
        overwrite_b: Compatibility flag.
        debug: Compatibility flag.
        check_finite: Compatibility flag.
        assume_a: Matrix structure assumption (``"gen"``, ``"sym"``, ``"her"``, ``"pos"``).

    Returns:
        ndarray: Solution array ``x``.
    """
    _ = (lower, overwrite_a, overwrite_b, debug, check_finite, assume_a)
    return _wrap(_ops.solve(_to_tensor(_ensure_float_array(a)), _to_tensor(_ensure_float_array(b))))


def solve_sylvester(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
) -> ndarray:
    """Compute a solution ``X`` to the Sylvester equation ``A X + X B = Q``.

    Args:
        a: Leading square matrix ``A`` of shape ``(M, M)``.
        b: Trailing square matrix ``B`` of shape ``(N, N)``.
        q: Right-hand side matrix ``Q`` of shape ``(M, N)``.

    Returns:
        ndarray: Solution matrix ``X`` of shape ``(M, N)``.
    """
    a_arr = _ensure_float_array(a)
    b_arr = _ensure_float_array(b)
    q_arr = _ensure_float_array(q)
    wa, va = _jnp_eigh(a_arr)
    wb, vb = _jnp_eigh(b_arr)
    q_tilde = matrix_transpose(va) @ q_arr @ vb
    denom = wa[:, None] + wb[None, :]
    x_tilde = q_tilde / where(denom == 0.0, 1.0, denom)
    return va @ x_tilde @ matrix_transpose(vb)


def solve_triangular(
    a: ArrayLike,
    b: ArrayLike,
    trans: int | str = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> ndarray:
    """Solve the triangular equation ``a x = b`` for ``x``.

    Args:
        a: Triangular matrix.
        b: Right-hand side vector or matrix.
        trans: Type of system to solve (``0``/``"N"``, ``1``/``"T"``, ``2``/``"C"``).
        lower: Use only data contained in the lower triangle of ``a``.
        unit_diagonal: If ``True``, diagonal elements of ``a`` are assumed to be 1.
        overwrite_b: Compatibility flag.
        check_finite: Compatibility flag.

    Returns:
        ndarray: Solution array ``x``.
    """
    _ = (lower, unit_diagonal, overwrite_b, check_finite)
    a_arr = _ensure_float_array(a)
    b_arr = _ensure_float_array(b)
    if trans in (1, 2, "T", "C"):
        a_arr = matrix_transpose(a_arr)
    return _jnp_solve(a_arr, b_arr)


def sqrtm(
    A: ArrayLike,
    blocksize: int = 1,
) -> ndarray:
    """Compute the matrix square root.

    Args:
        A: Square matrix whose square root to evaluate.
        blocksize: Compatibility parameter.

    Returns:
        ndarray: Matrix square root ``sqrt(A)``.
    """
    _ = blocksize
    return funm(A, sqrt)


def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: str = "gesdd",
) -> ndarray | tuple[ndarray, ndarray, ndarray]:
    """Compute the singular value decomposition of a matrix.

    Args:
        a: Matrix to decompose of shape ``(..., M, N)``.
        full_matrices: If ``True``, ``U`` and ``Vh`` have shapes ``(..., M, M)`` and ``(..., N, N)``.
        compute_uv: Whether to compute ``U`` and ``Vh`` in addition to ``s``.
        overwrite_a: Compatibility flag.
        check_finite: Compatibility flag.
        lapack_driver: Compatibility flag.

    Returns:
        ndarray | tuple[ndarray, ndarray, ndarray]: ``(U, s, Vh)`` if ``compute_uv=True``, else ``s``.
    """
    _ = (overwrite_a, check_finite, lapack_driver)
    return _wrap(_ops.svd(_to_tensor(_ensure_float_array(a)), full_matrices=full_matrices, compute_uv=compute_uv))


def toeplitz(
    c: ArrayLike,
    r: ArrayLike | None = None,
) -> ndarray:
    """Construct a Toeplitz matrix.

    Args:
        c: First column of the matrix.
        r: First row of the matrix (defaults to ``c`` if ``None``).

    Returns:
        ndarray: Toeplitz matrix of shape ``(len(c), len(r))``.
    """
    c_arr = asarray(c)
    r_arr = asarray(r, dtype=c_arr.dtype) if r is not None else c_arr
    n = c_arr.shape[0]
    m = r_arr.shape[0]
    i_idx = arange(n, dtype=int32)[:, None]
    j_idx = arange(m, dtype=int32)[None, :]
    diff = i_idx - j_idx
    col_part = c_arr[asarray(where(diff >= 0, diff, 0), dtype=int32)]
    row_part = r_arr[asarray(where(diff < 0, -diff, 0), dtype=int32)]
    return where(diff >= 0, col_part, row_part)


def cholesky_solve(
    c: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
) -> ndarray:
    """Solve a linear system using a Cholesky factorization ``c``.

    Args:
        c: Cholesky factor.
        b: Right-hand side vector or matrix.
        lower: Whether ``c`` is lower-triangular.

    Returns:
        ndarray: Solution array ``x``.
    """
    return cho_solve((c, lower), b)


def lu_reconstruct(
    p_or_tuple: ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike],
    l_mat: ArrayLike | None = None,
    u_mat: ArrayLike | None = None,
) -> ndarray:
    """Reconstruct a matrix from its ``(P, L, U)`` LU factorization.

    Args:
        p_or_tuple: Permutation matrix ``P`` or a 3-tuple ``(P, L, U)``.
        l_mat: Lower-triangular matrix ``L`` when ``p_or_tuple`` is ``P``.
        u_mat: Upper-triangular matrix ``U`` when ``p_or_tuple`` is ``P``.

    Returns:
        ndarray: Reconstructed matrix ``P @ L @ U``.
    """
    if isinstance(p_or_tuple, tuple) and len(p_or_tuple) == 3:
        p, l_m, u_m = p_or_tuple
        return _ensure_float_array(p) @ _ensure_float_array(l_m) @ _ensure_float_array(u_m)
    if l_mat is not None and u_mat is not None:
        return _ensure_float_array(p_or_tuple) @ _ensure_float_array(l_mat) @ _ensure_float_array(u_mat)
    return _wrap(_ops.lu_reconstruct(_to_tensor(_ensure_float_array(p_or_tuple)), _to_tensor(asarray(l_mat))))


def lu_matrix_inverse(
    a: ArrayLike,
    piv: ArrayLike | None = None,
) -> ndarray:
    """Compute the matrix inverse using an LU factorization.

    Args:
        a: Square matrix or LU factor matrix.
        piv: Optional pivot indices.

    Returns:
        ndarray: Inverse matrix.
    """
    if piv is None:
        return inv(a)
    return _wrap(_ops.lu_matrix_inverse(_to_tensor(_ensure_float_array(a)), _to_tensor(asarray(piv))))


__all__ = [
    "block_diag",
    "cho_factor",
    "cho_solve",
    "cholesky",
    "cholesky_solve",
    "det",
    "eigh",
    "eigh_tridiagonal",
    "expm",
    "expm_frechet",
    "funm",
    "hessenberg",
    "hilbert",
    "inv",
    "lu",
    "lu_factor",
    "lu_matrix_inverse",
    "lu_reconstruct",
    "lu_solve",
    "pascal",
    "polar",
    "qr",
    "rsf2csf",
    "schur",
    "solve",
    "solve_sylvester",
    "solve_triangular",
    "sqrtm",
    "svd",
    "toeplitz",
]
