"""Linear algebra decompositions and matrix solver proxy operations."""

from __future__ import annotations

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.ops.base import OpDef, get_op

cholesky_solve = _original_ops.cholesky_solve
corrcoef = _original_ops.corrcoef
cov = _original_ops.cov
cross = _original_ops.cross
dot = _original_ops.dot
dot_general = _original_ops.dot_general
eigh_tridiagonal = _original_ops.eigh_tridiagonal
einsum = _original_ops.einsum
expm = _original_ops.expm
inner = _original_ops.inner
lstsq = _original_ops.lstsq
lu = _original_ops.lu
lu_matrix_inverse = _original_ops.lu_matrix_inverse
lu_reconstruct = _original_ops.lu_reconstruct
lu_solve = _original_ops.lu_solve
matmul = _original_ops.matmul
matrix_norm = _original_ops.matrix_norm
matrix_rank = _original_ops.matrix_rank
matrix_transpose = _original_ops.matrix_transpose
multi_dot = _original_ops.multi_dot
norm = _original_ops.norm
outer = _original_ops.outer
polar = _original_ops.polar
sqrtm = _original_ops.sqrtm
svdvals = _original_ops.svdvals
tensordot = _original_ops.tensordot
tensorinv = _original_ops.tensorinv
tensorsolve = _original_ops.tensorsolve
trace = _original_ops.trace
triangular_solve = _original_ops.triangular_solve
tridiagonal_solve = _original_ops.tridiagonal_solve
vdot = _original_ops.vdot
vecdot = _original_ops.vecdot
vector_norm = _original_ops.vector_norm

block_diag: OpDef = get_op("BlockDiag")()
cholesky: OpDef = get_op("Cholesky")()
det: OpDef = get_op("Det")()
eig: OpDef = get_op("Eig")()
eigh: OpDef = get_op("Eigh")()
eigvals: OpDef = get_op("Eigvals")()
eigvalsh: OpDef = get_op("Eigvalsh")()
hessenberg: OpDef = get_op("Hessenberg")()
householder_product: OpDef = get_op("HouseholderProduct")()
inv: OpDef = get_op("Inv")()
lu_factor: OpDef = get_op("LuFactor")()
lu_pivots_to_permutation: OpDef = get_op("LuPivotsToPermutation")()
matrix_power: OpDef = get_op("MatrixPower")()
pinv: OpDef = get_op("Pinv")()
qdwh: OpDef = get_op("Qdwh")()
qr: OpDef = get_op("Qr")()
schur: OpDef = get_op("Schur")()
slogdet: OpDef = get_op("Slogdet")()
solve: OpDef = get_op("Solve")()
svd: OpDef = get_op("Svd")()
tridiagonal: OpDef = get_op("Tridiagonal")()

__all__ = [
    "block_diag",
    "cholesky",
    "cholesky_solve",
    "corrcoef",
    "cov",
    "cross",
    "det",
    "dot",
    "dot_general",
    "eig",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvalsh",
    "einsum",
    "expm",
    "hessenberg",
    "householder_product",
    "inner",
    "inv",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_matrix_inverse",
    "lu_pivots_to_permutation",
    "lu_reconstruct",
    "lu_solve",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "multi_dot",
    "norm",
    "outer",
    "pinv",
    "polar",
    "qdwh",
    "qr",
    "schur",
    "slogdet",
    "solve",
    "sqrtm",
    "svd",
    "svdvals",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "trace",
    "triangular_solve",
    "tridiagonal",
    "tridiagonal_solve",
    "vdot",
    "vecdot",
    "vector_norm",
]
