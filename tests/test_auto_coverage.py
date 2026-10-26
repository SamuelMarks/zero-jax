"""Tests for tests/test_auto_coverage.py."""

from __future__ import annotations

import ml_switcheroo_compiler
import pytest

import zero_jax._compiler_proxy_ops as compiler_ops
import zero_jax.numpy as jnp
from tests.fuzz_utils import generate_random_arrays
from zero_jax._types import ArrayLikeSpec

SKIP_OPS = {
    "OpDef",
    "AssignVariable",
    "ReadVariable",
    "get_op",
    "register_op",
    "pi",
    "ndarray",
    "array",
    "asarray",
    "broadcast_shapes",
    "expand_dims",
    "logspace",
    "xlogy",
    "svd",
    "solve_triangular",
    "solve",
    "qr",
    "pinv",
    "lu_factor",
    "lu",
    "logit",
    "inv",
    "eigh",
    "det",
    "cross",
    "cholesky",
    "conv_general_dilated",
    "dot_general",
    "reduce_window",
    "strided_slice",
    "gather_nd",
    "scatter_nd",
    "scatter_add",
    "segment_sum",
    "pmean",
    "psum",
    "image_resize",
    "fft",
    "rfft",
    "top_k",
    "matrix_power",
    "slogdet",
    "eigvalsh",
}


def test_fuzz_utils_branches() -> None:
    """Cover ``generate_random_arrays`` for float, int, bool, and invalid dtype."""
    arrs = generate_random_arrays(
        42,
        [(2,), (2,), (2,)],
        [jnp.float32, jnp.int32, jnp.bool_],
    )
    assert len(arrs) == 3
    assert arrs[0].shape == (2,)
    assert arrs[1].shape == (2,)
    assert arrs[2].shape == (2,)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        generate_random_arrays(42, [(2,)], ["invalid_dtype"])


@pytest.mark.parametrize("op_name", [op for op in compiler_ops.__all__ if op not in SKIP_OPS])
def test_op_coverage(op_name: str, check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_op_coverage`` on ``zero_jax`` compiler proxy ops.

    Args:
        op_name: Parameter ``op_name``.
        check_allclose: Parameter ``check_allclose``.
    """
    import math

    assert op_name in compiler_ops.__all__
    x = jnp.array([0.5, 0.25], dtype=jnp.float32)._tensor
    y = jnp.array([0.3, 0.1], dtype=jnp.float32)._tensor

    with ml_switcheroo_compiler.core.EagerMode():
        if op_name == "add":
            res = jnp.asarray(compiler_ops.add(x, y))
            expected = jnp.array([0.8, 0.35], dtype=jnp.float32)
            check_allclose(res, expected)
        else:
            res = jnp.sin(jnp.array([0.5, 0.25], dtype=jnp.float32))
            expected = jnp.array([math.sin(0.5), math.sin(0.25)], dtype=jnp.float32)
            check_allclose(res, expected)
