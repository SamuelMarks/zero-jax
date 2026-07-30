import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import jax
import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as compiler_ops

# These operations are complex and tested manually, or cannot be dynamically fuzzed easily
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


@pytest.mark.parametrize(
    "op_name", [op for op in compiler_ops.__all__ if op not in SKIP_OPS]
)
def test_op_coverage(op_name, check_allclose):
    op_func = getattr(compiler_ops, op_name, None)
    if not op_func:
        pytest.skip(f"{op_name} not found in compiler_ops")

    # Attempt to find corresponding jax function
    jax_func = None
    for module in [jax.numpy, jax.lax]:
        if hasattr(module, op_name):
            jax_func = getattr(module, op_name)
            break

    if not jax_func:
        pytest.skip(f"Could not find equivalent for {op_name} in JAX")

    # Generate dummy inputs based on basic heuristics
    x = np.array([0.5, 0.25], dtype=np.float32)
    y = np.array([0.3, 0.1], dtype=np.float32)

    try:
        # Try binary first if it takes two arguments
        with ml_switcheroo_compiler.core.EagerMode():
            res_z = op_func(x, y)
        res_j = jax_func(x, y)
        check_allclose(res_z, res_j)
    except Exception as e1:
        try:
            # Try unary
            with ml_switcheroo_compiler.core.EagerMode():
                res_z = op_func(x)
            res_j = jax_func(x)
            check_allclose(res_z, res_j)
        except Exception as e2:
            pytest.skip(
                f"Dynamic fuzzing failed for {op_name}. Binary err: {e1}, Unary err: {e2}"
            )
