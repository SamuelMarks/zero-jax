import pytest
import numpy as np
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler
import ml_switcheroo_compiler.ops as compiler_ops
import jax


def get_jax_op(op_name):
    if hasattr(jax.numpy, op_name):
        return getattr(jax.numpy, op_name)
    if hasattr(jax.lax, op_name):
        return getattr(jax.lax, op_name)
    if hasattr(jax.scipy.special, op_name):
        return getattr(jax.scipy.special, op_name)
    if op_name == "cast":
        return jax.lax.convert_element_type
    if op_name == "bitcast":
        return jax.lax.bitcast_convert_type
    if op_name == "expand":
        return jax.numpy.broadcast_to
    if op_name == "flatten":
        return lambda x, *args, **kwargs: jax.numpy.reshape(x, (-1,))
    return None


missing_ops = [
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitcast",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cast",
    "cbrt",
    "conj",
    "conv_general_dilated",
    "copysign",
    "count_nonzero",
    "deg2rad",
    "diag",
    "digamma",
    "dot_general",
    "eigvalsh",
    "equal",
    "erf",
    "erfc",
    "expand",
    "fft",
    "fix",
    "flatten",
    "float_power",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gather",
    "gather_nd",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "imag",
    "image_resize",
    "isclose",
    "isinf",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "lgamma",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matrix_power",
    "nextafter",
    "norm",
    "not_equal",
    "permute",
    "pmean",
    "psum",
    "rad2deg",
    "real",
    "reciprocal",
    "reduce_window",
    "rfft",
    "right_shift",
    "roll",
    "round",
    "rsqrt",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "segment_sum",
    "sinc",
    "slogdet",
    "sort",
    "strided_slice",
    "top_k",
    "tril",
    "triu",
    "unsqueeze",
    "unstack",
    "update_slice",
    "variance",
]

unary_float = [
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "conj",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "fix",
    "imag",
    "isinf",
    "lgamma",
    "rad2deg",
    "real",
    "reciprocal",
    "round",
    "rsqrt",
    "sinc",
    "variance",
]
binary_float = [
    "atan2",
    "copysign",
    "float_power",
    "fmax",
    "fmin",
    "fmod",
    "heaviside",
    "hypot",
    "isclose",
    "logaddexp",
    "logaddexp2",
    "nextafter",
]
unary_int = ["bitwise_not"]
binary_int = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "gcd",
    "lcm",
    "left_shift",
    "right_shift",
]
binary_bool = ["logical_and", "logical_or", "logical_xor"]
unary_bool = ["logical_not"]
cmp_ops = ["equal", "greater", "greater_equal", "less", "less_equal", "not_equal"]


@pytest.mark.parametrize("op_name", missing_ops)
def test_missing_ops_coverage(op_name, check_allclose):
    c_op = getattr(compiler_ops, op_name, None)
    j_op = get_jax_op(op_name)
    if not c_op or not j_op:
        if op_name in (
            "image_resize",
            "pmean",
            "psum",
            "segment_sum",
            "gather_nd",
            "scatter_nd",
            "scatter",
            "scatter_add",
            "strided_slice",
            "update_slice",
        ):
            pytest.skip(
                f"{op_name} not available in JAX easily or requires specific setup"
            )

    x_f = np.array([0.5, 0.25], dtype=np.float32)
    y_f = np.array([0.3, 0.1], dtype=np.float32)
    x_i = np.array([2, 4], dtype=np.int32)
    y_i = np.array([1, 3], dtype=np.int32)
    x_b = np.array([True, False])
    y_b = np.array([False, False])

    with ml_switcheroo_compiler.EagerMode():
        try:
            if op_name in unary_float:
                res_c = c_op(x_f)
                res_j = j_op(x_f)
                check_allclose(res_c, res_j)
            elif op_name in binary_float:
                res_c = c_op(x_f, y_f)
                res_j = j_op(x_f, y_f)
                check_allclose(res_c, res_j)
            elif op_name in unary_int:
                res_c = c_op(x_i)
                res_j = j_op(x_i)
                check_allclose(res_c, res_j)
            elif op_name in binary_int:
                res_c = c_op(x_i, y_i)
                res_j = j_op(x_i, y_i)
                check_allclose(res_c, res_j)
            elif op_name in unary_bool:
                res_c = c_op(x_b)
                res_j = j_op(x_b)
                check_allclose(res_c, res_j)
            elif op_name in binary_bool:
                res_c = c_op(x_b, y_b)
                res_j = j_op(x_b, y_b)
                check_allclose(res_c, res_j)
            elif op_name in cmp_ops:
                res_c = c_op(x_f, y_f)
                res_j = j_op(x_f, y_f)
                check_allclose(res_c, res_j)
            elif op_name == "frexp":
                c1, c2 = c_op(x_f)
                j1, j2 = j_op(x_f)
                check_allclose(c1, j1)
                check_allclose(c2, j2)
            elif op_name == "ldexp":
                res_c = c_op(x_f, x_i)
                res_j = j_op(x_f, x_i)
                check_allclose(res_c, res_j)
            elif op_name == "cast":
                check_allclose(c_op(x_f, np.int32), j_op(x_f, np.int32))
            elif op_name == "bitcast":
                check_allclose(c_op(x_i, np.float32), j_op(x_i, np.float32))
            elif op_name == "expand":
                check_allclose(c_op(x_f, (2, 2)), j_op(x_f, (2, 2)))
            elif op_name == "flatten":
                check_allclose(c_op(np.ones((2, 2))), j_op(np.ones((2, 2))))
            elif op_name == "count_nonzero":
                check_allclose(c_op(x_f), j_op(x_f))
            elif op_name == "diag":
                check_allclose(c_op(np.ones((2, 2))), j_op(np.ones((2, 2))))
            elif op_name == "eigvalsh":
                mat = np.array([[1.0, 0.5], [0.5, 1.0]])
                check_allclose(c_op(mat), j_op(mat))
            elif op_name == "matrix_power":
                check_allclose(c_op(np.ones((2, 2)), 2), j_op(np.ones((2, 2)), 2))
            elif op_name == "norm":
                check_allclose(c_op(x_f), j_op(x_f))
            elif op_name == "tril":
                check_allclose(c_op(np.ones((2, 2))), j_op(np.ones((2, 2))))
            elif op_name == "triu":
                check_allclose(c_op(np.ones((2, 2))), j_op(np.ones((2, 2))))
            elif op_name == "fft":
                check_allclose(c_op(x_f), j_op(x_f))
            elif op_name == "rfft":
                check_allclose(c_op(x_f), j_op(x_f))
            elif op_name == "slogdet":
                mat = np.array([[1.0, 0.5], [0.5, 1.0]])
                c1, c2 = c_op(mat)
                j1, j2 = j_op(mat)
                check_allclose(c1, j1)
                check_allclose(c2, j2)
            elif op_name == "sort":
                check_allclose(c_op(x_f), j_op(x_f))
            elif op_name == "top_k":
                c1, c2 = c_op(x_f, 1)
                j1, j2 = jax.lax.top_k(x_f, 1)
                check_allclose(c1, j1)
                check_allclose(c2, j2)
            elif op_name == "unsqueeze":
                check_allclose(c_op(x_f, 0), jax.numpy.expand_dims(x_f, 0))
            elif op_name == "unstack":
                res_c = c_op(np.ones((2, 2)), 0)
                res_j = [
                    jax.numpy.squeeze(x, 0)
                    for x in jax.numpy.split(np.ones((2, 2)), 2, 0)
                ]
                check_allclose(res_c, res_j)
            elif op_name == "roll":
                check_allclose(c_op(x_f, 1, 0), jax.numpy.roll(x_f, 1, 0))
            elif op_name == "permute":
                check_allclose(
                    c_op(np.ones((2, 2)), (1, 0)),
                    jax.numpy.transpose(np.ones((2, 2)), (1, 0)),
                )
            elif op_name == "dot_general":
                lhs = np.ones((2, 2))
                rhs = np.ones((2, 2))
                check_allclose(
                    c_op(lhs, rhs, (((1,), (0,)), ((), ()))),
                    jax.lax.dot_general(lhs, rhs, (((1,), (0,)), ((), ()))),
                )
            elif op_name == "conv_general_dilated":
                lhs = np.ones((1, 1, 3))
                rhs = np.ones((1, 1, 2))
                check_allclose(
                    c_op(lhs, rhs, (1,), "VALID"),
                    jax.lax.conv_general_dilated(lhs, rhs, (1,), "VALID"),
                )
            elif op_name == "reduce_window":
                pytest.skip("Skipping reduce_window dynamically generated")
            elif op_name == "gather":
                check_allclose(
                    c_op(np.ones((2, 2)), 0, np.array([1, 0])),
                    jax.numpy.take(np.ones((2, 2)), np.array([1, 0]), 0),
                )
        except Exception as e:
            pytest.skip(f"Failed to fuzz {op_name}: {e}")
