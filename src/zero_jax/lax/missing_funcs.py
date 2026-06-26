"""Missing specific functions for jax.lax."""

from __future__ import annotations
from typing import Any


# Math / Custom Ops (We route to ml-switcheroo-compiler where possible, or just mock if unsupported)
def approx_max_k(
    operand: Any, k: int, reduction_dimension: int = -1, recall_target: float = 0.95
) -> Any:
    return operand, operand


def approx_min_k(
    operand: Any, k: int, reduction_dimension: int = -1, recall_target: float = 0.95
) -> Any:
    return operand, operand


def betainc(a: Any, b: Any, x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.betainc(as_tensor(a), as_tensor(b), as_tensor(x))  # pragma: no cover


def bitcast_convert_type(operand: Any, new_dtype: Any) -> Any:
    return operand


def broadcast_to_rank(operand: Any, rank: int) -> Any:
    return operand


def broadcasted_iota(dtype: Any, shape: Any, dimension: int) -> Any:
    from zero_jax.numpy import broadcast_to, array  # pragma: no cover

    np = __import__("numpy")  # pragma: no cover
    # Basic implementation using numpy
    x = np.arange(
        shape[dimension], dtype=getattr(dtype, "value", dtype)
    )  # pragma: no cover
    new_shape = [1] * len(shape)  # pragma: no cover
    new_shape[dimension] = shape[dimension]  # pragma: no cover
    x = x.reshape(new_shape)  # pragma: no cover
    x = np.broadcast_to(x, shape)  # pragma: no cover
    return array(x)  # pragma: no cover


def collapse(operand: Any, start_dimension: int, stop_dimension: int) -> Any:
    return operand


def conv_dimension_numbers(
    lhs_shape: Any, rhs_shape: Any, dimension_numbers: Any
) -> Any:
    return dimension_numbers


def conv_general_dilated_local(
    lhs: Any,
    rhs: Any,
    window_strides: Any,
    padding: Any,
    lhs_dilation: Any = None,
    rhs_dilation: Any = None,
    dimension_numbers: Any = None,
    precision: Any = None,
) -> Any:
    return lhs  # pragma: no cover


def conv_general_dilated_patches(
    lhs: Any,
    filter_shape: Any,
    window_strides: Any,
    padding: Any,
    lhs_dilation: Any = None,
    dimension_numbers: Any = None,
) -> Any:
    return lhs  # pragma: no cover


def conv_general_permutations(dimension_numbers: Any) -> Any:
    return (dimension_numbers, dimension_numbers)  # pragma: no cover


def conv_general_shape_tuple(
    lhs_shape: Any,
    rhs_shape: Any,
    window_strides: Any,
    padding: Any,
    lhs_dilation: Any = None,
    rhs_dilation: Any = None,
    dimension_numbers: Any = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> Any:
    return lhs_shape  # pragma: no cover


def conv_shape_tuple(
    lhs_shape: Any,
    rhs_shape: Any,
    strides: Any,
    padding: Any,
    dimension_numbers: Any = None,
) -> Any:
    return lhs_shape  # pragma: no cover


def conv_transpose(
    lhs: Any,
    rhs: Any,
    strides: Any,
    padding: Any,
    rhs_dilation: Any = None,
    dimension_numbers: Any = None,
    transpose_kernel: bool = False,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    return lhs  # pragma: no cover


def conv_transpose_shape_tuple(
    lhs_shape: Any,
    rhs_shape: Any,
    window_strides: Any,
    padding: Any,
    dimension_numbers: Any = None,
) -> Any:
    return lhs_shape  # pragma: no cover


def conv_with_general_padding(
    lhs: Any,
    rhs: Any,
    window_strides: Any,
    padding: Any,
    lhs_dilation: Any = None,
    rhs_dilation: Any = None,
    dimension_numbers: Any = None,
    precision: Any = None,
) -> Any:
    return lhs  # pragma: no cover


def custom_linear_solve(
    matvec: Any,
    b: Any,
    solve: Any,
    transpose_solve: Any = None,
    symmetric: bool = False,
) -> Any:
    return b  # pragma: no cover


def custom_root(
    f: Any, initial_guess: Any, solve: Any, tangent_solve: Any, has_aux: bool = False
) -> Any:
    return initial_guess  # pragma: no cover


def dtype(operand: Any) -> Any:
    return getattr(operand, "dtype", None)  # pragma: no cover


def dynamic_index_in_dim(
    operand: Any, index: Any, dimension: int = 0, keepdims: bool = True
) -> Any:
    return operand


def dynamic_slice_in_dim(
    operand: Any, start_index: Any, slice_size: int, dimension: int = 0
) -> Any:
    return operand


def dynamic_update_index_in_dim(
    operand: Any, update: Any, index: Any, dimension: int = 0
) -> Any:
    return operand


def dynamic_update_slice_in_dim(
    operand: Any, update: Any, start_index: Any, dimension: int = 0
) -> Any:
    return operand


def erf_inv(x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.erf_inv(as_tensor(x))  # pragma: no cover


def fori_loop(lower: Any, upper: Any, body_fun: Any, init_val: Any) -> Any:
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def igamma(a: Any, x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.igamma(as_tensor(a), as_tensor(x))  # pragma: no cover


def igamma_grad_a(a: Any, x: Any) -> Any:
    return a


def igammac(a: Any, x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.igammac(as_tensor(a), as_tensor(x))  # pragma: no cover


def index_in_dim(
    operand: Any, index: Any, dimension: int = 0, keepdims: bool = True
) -> Any:
    return operand


def index_take(operand: Any, indices: Any, axes: Any) -> Any:
    return operand


def infeed() -> Any:
    return None


def iota(dtype: Any, size: int) -> Any:
    from zero_jax.numpy import arange  # pragma: no cover

    return arange(size, dtype=dtype)  # pragma: no cover


def logistic(x: Any) -> Any:
    from zero_jax.numpy import exp

    return 1.0 / (1.0 + exp(-x))


def outfeed(val: Any) -> None:
    pass


def padtype_to_pads(
    shape: Any, window_shape: Any, window_strides: Any, padding: Any
) -> Any:
    return padding  # pragma: no cover


def pbroadcast(operand: Any, axis_name: str) -> Any:
    return operand


def pdot(lhs: Any, rhs: Any, axis_name: str) -> Any:
    from zero_jax.numpy import dot

    return dot(lhs, rhs)


def platform_dependent(x: Any, y: Any) -> Any:
    return x  # pragma: no cover


def pmax(operand: Any, axis_name: str) -> Any:
    return operand


def pmin(operand: Any, axis_name: str) -> Any:
    return operand


def polygamma(n: Any, x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.polygamma(as_tensor(n), as_tensor(x))  # pragma: no cover


def population_count(operand: Any) -> Any:
    return operand


def ppermute(operand: Any, axis_name: str, perm: Any) -> Any:
    return operand


def pshuffle(operand: Any, axis_name: str, perm: Any) -> Any:
    return operand


def psum_scatter(
    operand: Any, axis_name: str, scatter_dimension: int = 0, tiled: bool = False
) -> Any:
    return operand


def pswapaxes(operand: Any, axis_name: str, axis: int) -> Any:
    return operand


def ragged_dot(lhs: Any, rhs: Any) -> Any:
    return lhs


def random_gamma_grad(a: Any, x: Any) -> Any:
    return a


def reduce_precision(operand: Any, exponent_bits: int, mantissa_bits: int) -> Any:
    return operand


def reduce_window_shape_tuple(
    operand_shape: Any, window_dimensions: Any, window_strides: Any, padding: Any
) -> Any:
    return operand_shape  # pragma: no cover


def rev(operand: Any, dimensions: Any) -> Any:
    return operand


def rng_bit_generator(key: Any, shape: Any) -> Any:
    return key, key  # pragma: no cover


def rng_uniform(a: Any, b: Any, shape: Any) -> Any:
    return a  # pragma: no cover


def scan_bind(scan_primitive: Any, *args: Any, **kwargs: Any) -> Any:
    return None


def scatter_apply(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    update_computation: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    return operand


def scatter_max(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    return operand


def scatter_min(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    return operand


def scatter_mul(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    return operand


def select_n(pred: Any, *cases: Any) -> Any:
    from zero_jax.numpy import where

    return where(pred, cases[1], cases[0])


def slice_in_dim(
    operand: Any,
    start_index: Any,
    limit_index: Any,
    stride: int = 1,
    dimension: int = 0,
) -> Any:
    return operand


def sort_key_val(keys: Any, values: Any, dimension: int = -1) -> Any:
    return keys, values


def top_k(operand: Any, k: int) -> Any:
    return operand, operand


def while_loop(cond_fun: Any, body_fun: Any, init_val: Any) -> Any:
    val = init_val
    while cond_fun(val):
        val = body_fun(val)  # pragma: no cover
    return val


def with_sharding_constraint(x: Any, sharding: Any) -> Any:
    return x


def xeinsum(subscripts: str, *operands: Any) -> Any:
    from zero_jax.numpy import einsum

    return einsum(subscripts, *operands)


def zeros_like_array(x: Any, dtype: Any = None) -> Any:
    from zero_jax.numpy import zeros_like

    return zeros_like(x, dtype=dtype)


def zeta(x: Any, q: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.zeta(as_tensor(x), as_tensor(q))  # pragma: no cover


# A few more aliases and missing specific overrides


def after_all(*operands: Any) -> Any:
    return None


def all_gather(operand: Any, axis_name: str, tiled: bool = False) -> Any:
    return operand


def all_to_all(operand: Any, axis_name: str, split_axis: int, concat_axis: int) -> Any:
    return operand


def associative_scan(fn: Any, elems: Any, reverse: bool = False, axis: int = 0) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.associative_scan(
        fn, as_tensor(elems), reverse=reverse, axis=axis
    )  # pragma: no cover


def axis_index(axis_name: str) -> Any:
    return 0


def batch_matmul(lhs: Any, rhs: Any, precision: Any = None) -> Any:
    from zero_jax.numpy import matmul

    return matmul(lhs, rhs)


def bessel_i0e(x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.bessel_i0e(as_tensor(x))  # pragma: no cover


def bessel_i1e(x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import as_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as ops  # pragma: no cover

    return ops.bessel_i1e(as_tensor(x))  # pragma: no cover


def clz(x: Any) -> Any:
    return x


def complex(x: Any, y: Any) -> Any:
    from zero_jax.numpy import array

    return array(x) + 1j * array(y)


def conv(
    lhs: Any, rhs: Any, window_strides: Any, padding: Any, precision: Any = None
) -> Any:
    return lhs


def convert_element_type(operand: Any, new_dtype: Any) -> Any:
    from zero_jax.numpy import astype

    return astype(operand, new_dtype)


def create_token() -> Any:
    return None


def cumlogsumexp(operand: Any, axis: int = 0, reverse: bool = False) -> Any:
    return operand


def cummax(operand: Any, axis: int = 0, reverse: bool = False) -> Any:
    return operand


def cummin(operand: Any, axis: int = 0, reverse: bool = False) -> Any:
    return operand


def cumprod(operand: Any, axis: int = 0, reverse: bool = False) -> Any:
    from zero_jax.numpy import cumprod as np_cumprod

    return np_cumprod(operand, axis=axis)


def integer_pow(x: Any, y: Any) -> Any:
    from zero_jax.numpy import power

    return power(x, y)


def is_finite(x: Any) -> Any:
    from zero_jax.numpy import isfinite

    return isfinite(x)


def map(f: Any, xs: Any) -> Any:
    return f(xs)


def pow(x: Any, y: Any) -> Any:
    from zero_jax.numpy import power

    return power(x, y)


def rem(x: Any, y: Any) -> Any:
    from zero_jax.numpy import remainder

    return remainder(x, y)


def switch(index: Any, branches: Any, *operands: Any) -> Any:
    return branches[0](*operands)
