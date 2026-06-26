"""Missing specific functions for jax.lax."""

from __future__ import annotations
from typing import Any


def approx_max_k(
    operand: Any, k: int, reduction_dimension: int = -1, recall_target: float = 0.95
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.approx_max_k(_to_tensor(operand), k, reduction_dimension, recall_target)
    )


def approx_min_k(
    operand: Any, k: int, reduction_dimension: int = -1, recall_target: float = 0.95
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.approx_min_k(_to_tensor(operand), k, reduction_dimension, recall_target)
    )


def betainc(a: Any, b: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.betainc(_to_tensor(a), _to_tensor(b), _to_tensor(x)))


def bitcast_convert_type(operand: Any, new_dtype: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.bitcast_convert_type(_to_tensor(operand), _to_tensor(new_dtype)))


def broadcast_to_rank(operand: Any, rank: int) -> Any:
    return operand


def broadcasted_iota(dtype: Any, shape: Any, dimension: int) -> Any:
    from zero_jax.numpy import broadcast_to, array

    np = __import__("numpy")
    x = np.arange(shape[dimension], dtype=getattr(dtype, "value", dtype))
    new_shape = [1] * len(shape)
    new_shape[dimension] = shape[dimension]
    x = x.reshape(new_shape)
    x = np.broadcast_to(x, shape)
    return array(x)


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
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.conv_general_dilated_local(
            _to_tensor(lhs),
            _to_tensor(rhs),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(rhs_dilation),
            _to_tensor(dimension_numbers),
            _to_tensor(precision),
        )
    )


def conv_general_dilated_patches(
    lhs: Any,
    filter_shape: Any,
    window_strides: Any,
    padding: Any,
    lhs_dilation: Any = None,
    dimension_numbers: Any = None,
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.conv_general_dilated_patches(
            _to_tensor(lhs),
            _to_tensor(filter_shape),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(dimension_numbers),
        )
    )


def conv_general_permutations(dimension_numbers: Any) -> Any:
    return dimension_numbers, dimension_numbers


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
    return lhs_shape


def conv_shape_tuple(
    lhs_shape: Any,
    rhs_shape: Any,
    strides: Any,
    padding: Any,
    dimension_numbers: Any = None,
) -> Any:
    return lhs_shape


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
    return lhs


def conv_transpose_shape_tuple(
    lhs_shape: Any,
    rhs_shape: Any,
    window_strides: Any,
    padding: Any,
    dimension_numbers: Any = None,
) -> Any:
    return lhs_shape


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
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.conv_with_general_padding(
            _to_tensor(lhs),
            _to_tensor(rhs),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(rhs_dilation),
            _to_tensor(dimension_numbers),
            _to_tensor(precision),
        )
    )


def custom_linear_solve(
    matvec: Any,
    b: Any,
    solve: Any,
    transpose_solve: Any = None,
    symmetric: bool = False,
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.custom_linear_solve(
            _to_tensor(matvec),
            _to_tensor(b),
            _to_tensor(solve),
            _to_tensor(transpose_solve),
            symmetric,
        )
    )


def custom_root(
    f: Any, initial_guess: Any, solve: Any, tangent_solve: Any, has_aux: bool = False
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.custom_root(
            _to_tensor(f),
            _to_tensor(initial_guess),
            _to_tensor(solve),
            _to_tensor(tangent_solve),
            has_aux,
        )
    )


def dtype(operand: Any) -> Any:
    return getattr(operand, "dtype", None)


def dynamic_index_in_dim(
    operand: Any, index: Any, dimension: int = 0, keepdims: bool = True
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.dynamic_index_in_dim(
            _to_tensor(operand), _to_tensor(index), dimension, keepdims
        )
    )


def dynamic_slice_in_dim(
    operand: Any, start_index: Any, slice_size: int, dimension: int = 0
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.dynamic_slice_in_dim(
            _to_tensor(operand), _to_tensor(start_index), slice_size, dimension
        )
    )


def dynamic_update_index_in_dim(
    operand: Any, update: Any, index: Any, dimension: int = 0
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.dynamic_update_index_in_dim(
            _to_tensor(operand), _to_tensor(update), _to_tensor(index), dimension
        )
    )


def dynamic_update_slice_in_dim(
    operand: Any, update: Any, start_index: Any, dimension: int = 0
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.dynamic_update_slice_in_dim(
            _to_tensor(operand), _to_tensor(update), _to_tensor(start_index), dimension
        )
    )


def erf_inv(x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.erf_inv(_to_tensor(x)))


def fori_loop(lower: Any, upper: Any, body_fun: Any, init_val: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.fori_loop(
            _to_tensor(lower), _to_tensor(upper), body_fun, _to_tensor(init_val)
        )
    )


def igamma(a: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.igamma(_to_tensor(a), _to_tensor(x)))


def igamma_grad_a(a: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.igamma_grad_a(_to_tensor(a), _to_tensor(x)))


def igammac(a: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.igammac(_to_tensor(a), _to_tensor(x)))


def index_in_dim(
    operand: Any, index: Any, dimension: int = 0, keepdims: bool = True
) -> Any:
    return operand


def index_take(operand: Any, indices: Any, axes: Any) -> Any:
    return operand


def infeed() -> Any:
    return None


def iota(dtype: Any, size: int) -> Any:
    from zero_jax.numpy import arange

    return arange(size, dtype=dtype)


def logistic(x: Any) -> Any:
    from zero_jax.numpy import exp

    return 1.0 / (1.0 + exp(-x))


def outfeed(val: Any) -> None:
    pass


def padtype_to_pads(
    shape: Any, window_shape: Any, window_strides: Any, padding: Any
) -> Any:
    return padding


def pbroadcast(operand: Any, axis_name: str) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pbroadcast(_to_tensor(operand), axis_name))


def pdot(lhs: Any, rhs: Any, axis_name: str) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pdot(_to_tensor(lhs), _to_tensor(rhs), axis_name))


def platform_dependent(x: Any, y: Any) -> Any:
    return x


def pmax(operand: Any, axis_name: str) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pmax(_to_tensor(operand), axis_name))


def pmin(operand: Any, axis_name: str) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pmin(_to_tensor(operand), axis_name))


def polygamma(n: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.polygamma(_to_tensor(n), _to_tensor(x)))


def population_count(operand: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.population_count(_to_tensor(operand)))


def ppermute(operand: Any, axis_name: str, perm: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.ppermute(_to_tensor(operand), axis_name, _to_tensor(perm)))


def pshuffle(operand: Any, axis_name: str, perm: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pshuffle(_to_tensor(operand), axis_name, _to_tensor(perm)))


def psum_scatter(
    operand: Any, axis_name: str, scatter_dimension: int = 0, tiled: bool = False
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.psum_scatter(_to_tensor(operand), scatter_dimension, axis_name))


def pswapaxes(operand: Any, axis_name: str, axis: int) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pswapaxes(_to_tensor(operand), axis_name, axis))


def ragged_dot(lhs: Any, rhs: Any) -> Any:
    return lhs


def random_gamma_grad(a: Any, x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.random_gamma_grad(_to_tensor(a), _to_tensor(x)))


def reduce_precision(operand: Any, exponent_bits: int, mantissa_bits: int) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.reduce_precision(_to_tensor(operand), exponent_bits, mantissa_bits)
    )


def reduce_window_shape_tuple(
    operand_shape: Any, window_dimensions: Any, window_strides: Any, padding: Any
) -> Any:
    return operand_shape


def rev(operand: Any, dimensions: Any) -> Any:
    return operand


def rng_bit_generator(key: Any, shape: Any) -> Any:
    return key, key


def rng_uniform(a: Any, b: Any, shape: Any) -> Any:
    return a


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
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.scatter_apply(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(update_computation),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_max(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.scatter_max(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_min(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.scatter_min(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_mul(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.scatter_mul(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


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
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.slice_in_dim(
            _to_tensor(operand),
            _to_tensor(start_index),
            _to_tensor(limit_index),
            stride,
            dimension,
        )
    )


def sort_key_val(keys: Any, values: Any, dimension: int = -1) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.sort_key_val(_to_tensor(keys), _to_tensor(values), dimension))


def top_k(operand: Any, k: int) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.top_k(_to_tensor(operand), k))


def while_loop(cond_fun: Any, body_fun: Any, init_val: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.while_loop(cond_fun, body_fun, _to_tensor(init_val)))


def with_sharding_constraint(x: Any, sharding: Any) -> Any:
    return x


def xeinsum(subscripts: str, *operands: Any) -> Any:
    from zero_jax.numpy import einsum

    return einsum(subscripts, *operands)


def zeros_like_array(x: Any, dtype: Any = None) -> Any:
    from zero_jax.numpy import zeros_like

    return zeros_like(x, dtype=dtype)


def zeta(x: Any, q: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.zeta(_to_tensor(x), _to_tensor(q)))


def after_all(*operands: Any) -> Any:
    return None


def all_gather(operand: Any, axis_name: str, tiled: bool = False) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.all_gather(_to_tensor(operand), axis_name))


def all_to_all(operand: Any, axis_name: str, split_axis: int, concat_axis: int) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.all_to_all(_to_tensor(operand), split_axis, concat_axis, axis_name)
    )


def associative_scan(fn: Any, elems: Any, reverse: bool = False, axis: int = 0) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.associative_scan(fn, _to_tensor(elems), reverse, axis))


def axis_index(axis_name: str) -> Any:
    return 0


def batch_matmul(lhs: Any, rhs: Any, precision: Any = None) -> Any:
    from zero_jax.numpy import matmul

    return matmul(lhs, rhs)


def bessel_i0e(x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.bessel_i0e(_to_tensor(x)))


def bessel_i1e(x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.bessel_i1e(_to_tensor(x)))


def clz(x: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.clz(_to_tensor(x)))


def complex(x: Any, y: Any) -> Any:
    from zero_jax.numpy import array

    return array(x) + 1.0j * array(y)


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


__all__ = [
    "approx_max_k",
    "approx_min_k",
    "betainc",
    "bitcast_convert_type",
    "broadcast_to_rank",
    "broadcasted_iota",
    "collapse",
    "conv_dimension_numbers",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_general_permutations",
    "conv_general_shape_tuple",
    "conv_shape_tuple",
    "conv_transpose",
    "conv_transpose_shape_tuple",
    "conv_with_general_padding",
    "custom_linear_solve",
    "custom_root",
    "dtype",
    "dynamic_index_in_dim",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
    "erf_inv",
    "fori_loop",
    "igamma",
    "igamma_grad_a",
    "igammac",
    "index_in_dim",
    "index_take",
    "infeed",
    "iota",
    "logistic",
    "outfeed",
    "padtype_to_pads",
    "pbroadcast",
    "pdot",
    "platform_dependent",
    "pmax",
    "pmin",
    "polygamma",
    "population_count",
    "ppermute",
    "pshuffle",
    "psum_scatter",
    "pswapaxes",
    "ragged_dot",
    "random_gamma_grad",
    "reduce_precision",
    "reduce_window_shape_tuple",
    "rev",
    "rng_bit_generator",
    "rng_uniform",
    "scan_bind",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "select_n",
    "slice_in_dim",
    "sort_key_val",
    "top_k",
    "while_loop",
    "with_sharding_constraint",
    "xeinsum",
    "zeros_like_array",
    "zeta",
    "after_all",
    "all_gather",
    "all_to_all",
    "associative_scan",
    "axis_index",
    "batch_matmul",
    "bessel_i0e",
    "bessel_i1e",
    "clz",
    "complex",
    "conv",
    "convert_element_type",
    "create_token",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "integer_pow",
    "is_finite",
    "map",
    "pow",
    "rem",
    "switch",
]
