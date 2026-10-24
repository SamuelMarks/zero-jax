"""Primitive objects (`*_p`) for `jax.lax`."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


class MockPrimitive:
    """Represents a bindable JAX primitive descriptor (`*_p`)."""

    def __init__(self, name: str) -> None:
        """Initialize a ``MockPrimitive`` with its canonical primitive name.

        Args:
            name: The primitive name (e.g. ``"add_p"`` or ``"sin_p"``).
        """
        self.name = name
        self.multiple_results = False

    def bind(self, *args: ArrayLikeSpec, **params: ConfigValue) -> ArrayLikeSpec:
        """Bind arguments to this primitive by dispatching to the corresponding ``zero_jax.lax`` operation.

        Args:
            *args: Positional tensor or scalar operands.
            **params: Primitive keyword parameters.

        Returns:
            ArrayLikeSpec: Result of the primitive evaluation.
        """
        import zero_jax.lax as _lax

        op_name = self.name[:-2] if self.name.endswith("_p") else self.name
        lax_dict = _lax.__dict__
        if op_name in lax_dict and callable(lax_dict[op_name]):
            return lax_dict[op_name](*args, **params)
        return args[0] if args else None


abs_p = MockPrimitive("abs_p")
acos_p = MockPrimitive("acos_p")
acosh_p = MockPrimitive("acosh_p")
add_p = MockPrimitive("add_p")
after_all_p = MockPrimitive("after_all_p")
all_gather_p = MockPrimitive("all_gather_p")
all_to_all_p = MockPrimitive("all_to_all_p")
and_p = MockPrimitive("and_p")
approx_top_k_p = MockPrimitive("approx_top_k_p")
argmax_p = MockPrimitive("argmax_p")
argmin_p = MockPrimitive("argmin_p")
asin_p = MockPrimitive("asin_p")
asinh_p = MockPrimitive("asinh_p")
atan2_p = MockPrimitive("atan2_p")
atan_p = MockPrimitive("atan_p")
atanh_p = MockPrimitive("atanh_p")
axis_index_p = MockPrimitive("axis_index_p")
bessel_i0e_p = MockPrimitive("bessel_i0e_p")
bessel_i1e_p = MockPrimitive("bessel_i1e_p")
bitcast_convert_type_p = MockPrimitive("bitcast_convert_type_p")
broadcast_in_dim_p = MockPrimitive("broadcast_in_dim_p")
cbrt_p = MockPrimitive("cbrt_p")
ceil_p = MockPrimitive("ceil_p")
clamp_p = MockPrimitive("clamp_p")
clz_p = MockPrimitive("clz_p")
complex_p = MockPrimitive("complex_p")
concatenate_p = MockPrimitive("concatenate_p")
cond_p = MockPrimitive("cond_p")
conj_p = MockPrimitive("conj_p")
conv_general_dilated_p = MockPrimitive("conv_general_dilated_p")
convert_element_type_p = MockPrimitive("convert_element_type_p")
copy_p = MockPrimitive("copy_p")
cos_p = MockPrimitive("cos_p")
cosh_p = MockPrimitive("cosh_p")
create_token_p = MockPrimitive("create_token_p")
cumlogsumexp_p = MockPrimitive("cumlogsumexp_p")
cummax_p = MockPrimitive("cummax_p")
cummin_p = MockPrimitive("cummin_p")
cumprod_p = MockPrimitive("cumprod_p")
cumsum_p = MockPrimitive("cumsum_p")
device_put_p = MockPrimitive("device_put_p")
digamma_p = MockPrimitive("digamma_p")
div_p = MockPrimitive("div_p")
dot_general_p = MockPrimitive("dot_general_p")
dynamic_slice_p = MockPrimitive("dynamic_slice_p")
dynamic_update_slice_p = MockPrimitive("dynamic_update_slice_p")
eq_p = MockPrimitive("eq_p")
eq_to_p = MockPrimitive("eq_to_p")
erf_inv_p = MockPrimitive("erf_inv_p")
erf_p = MockPrimitive("erf_p")
erfc_p = MockPrimitive("erfc_p")
exp2_p = MockPrimitive("exp2_p")
exp_p = MockPrimitive("exp_p")
expm1_p = MockPrimitive("expm1_p")
fft_p = MockPrimitive("fft_p")
floor_p = MockPrimitive("floor_p")
gather_p = MockPrimitive("gather_p")
ge_p = MockPrimitive("ge_p")
gt_p = MockPrimitive("gt_p")
igamma_grad_a_p = MockPrimitive("igamma_grad_a_p")
igamma_p = MockPrimitive("igamma_p")
igammac_p = MockPrimitive("igammac_p")
imag_p = MockPrimitive("imag_p")
infeed_p = MockPrimitive("infeed_p")
integer_pow_p = MockPrimitive("integer_pow_p")
iota_p = MockPrimitive("iota_p")
is_finite_p = MockPrimitive("is_finite_p")
le_p = MockPrimitive("le_p")
lgamma_p = MockPrimitive("lgamma_p")
linear_solve_p = MockPrimitive("linear_solve_p")
log1p_p = MockPrimitive("log1p_p")
log_p = MockPrimitive("log_p")
logistic_p = MockPrimitive("logistic_p")
lt_p = MockPrimitive("lt_p")
max_p = MockPrimitive("max_p")
min_p = MockPrimitive("min_p")
mul_p = MockPrimitive("mul_p")
ne_p = MockPrimitive("ne_p")
neg_p = MockPrimitive("neg_p")
nextafter_p = MockPrimitive("nextafter_p")
not_p = MockPrimitive("not_p")
optimization_barrier_p = MockPrimitive("optimization_barrier_p")
or_p = MockPrimitive("or_p")
outfeed_p = MockPrimitive("outfeed_p")
pad_p = MockPrimitive("pad_p")
pmax_p = MockPrimitive("pmax_p")
pmin_p = MockPrimitive("pmin_p")
polygamma_p = MockPrimitive("polygamma_p")
population_count_p = MockPrimitive("population_count_p")
pow_p = MockPrimitive("pow_p")
ppermute_p = MockPrimitive("ppermute_p")
psum_p = MockPrimitive("psum_p")
random_gamma_grad_p = MockPrimitive("random_gamma_grad_p")
real_p = MockPrimitive("real_p")
reduce_and_p = MockPrimitive("reduce_and_p")
reduce_max_p = MockPrimitive("reduce_max_p")
reduce_min_p = MockPrimitive("reduce_min_p")
reduce_or_p = MockPrimitive("reduce_or_p")
reduce_p = MockPrimitive("reduce_p")
reduce_precision_p = MockPrimitive("reduce_precision_p")
reduce_prod_p = MockPrimitive("reduce_prod_p")
reduce_sum_p = MockPrimitive("reduce_sum_p")
reduce_window_max_p = MockPrimitive("reduce_window_max_p")
reduce_window_min_p = MockPrimitive("reduce_window_min_p")
reduce_window_p = MockPrimitive("reduce_window_p")
reduce_window_sum_p = MockPrimitive("reduce_window_sum_p")
reduce_xor_p = MockPrimitive("reduce_xor_p")
regularized_incomplete_beta_p = MockPrimitive("regularized_incomplete_beta_p")
rem_p = MockPrimitive("rem_p")
reshape_p = MockPrimitive("reshape_p")
rev_p = MockPrimitive("rev_p")
rng_bit_generator_p = MockPrimitive("rng_bit_generator_p")
rng_uniform_p = MockPrimitive("rng_uniform_p")
round_p = MockPrimitive("round_p")
rsqrt_p = MockPrimitive("rsqrt_p")
scan_p = MockPrimitive("scan_p")
scatter_add_p = MockPrimitive("scatter_add_p")
scatter_apply_p = MockPrimitive("scatter_apply_p")
scatter_max_p = MockPrimitive("scatter_max_p")
scatter_min_p = MockPrimitive("scatter_min_p")
scatter_mul_p = MockPrimitive("scatter_mul_p")
scatter_p = MockPrimitive("scatter_p")
select_n_p = MockPrimitive("select_n_p")
select_p = MockPrimitive("select_p")
shift_left_p = MockPrimitive("shift_left_p")
shift_right_arithmetic_p = MockPrimitive("shift_right_arithmetic_p")
shift_right_logical_p = MockPrimitive("shift_right_logical_p")
sign_p = MockPrimitive("sign_p")
sin_p = MockPrimitive("sin_p")
sinh_p = MockPrimitive("sinh_p")
slice_p = MockPrimitive("slice_p")
sort_p = MockPrimitive("sort_p")
split_p = MockPrimitive("split_p")
sqrt_p = MockPrimitive("sqrt_p")
square_p = MockPrimitive("square_p")
squeeze_p = MockPrimitive("squeeze_p")
stop_gradient_p = MockPrimitive("stop_gradient_p")
sub_p = MockPrimitive("sub_p")
tan_p = MockPrimitive("tan_p")
tanh_p = MockPrimitive("tanh_p")
top_k_p = MockPrimitive("top_k_p")
transpose_p = MockPrimitive("transpose_p")
while_p = MockPrimitive("while_p")
xor_p = MockPrimitive("xor_p")
zeta_p = MockPrimitive("zeta_p")

_primitives = [
    "abs_p",
    "acos_p",
    "acosh_p",
    "add_p",
    "after_all_p",
    "all_gather_p",
    "all_to_all_p",
    "and_p",
    "approx_top_k_p",
    "argmax_p",
    "argmin_p",
    "asin_p",
    "asinh_p",
    "atan2_p",
    "atan_p",
    "atanh_p",
    "axis_index_p",
    "bessel_i0e_p",
    "bessel_i1e_p",
    "bitcast_convert_type_p",
    "broadcast_in_dim_p",
    "cbrt_p",
    "ceil_p",
    "clamp_p",
    "clz_p",
    "complex_p",
    "concatenate_p",
    "cond_p",
    "conj_p",
    "conv_general_dilated_p",
    "convert_element_type_p",
    "copy_p",
    "cos_p",
    "cosh_p",
    "create_token_p",
    "cumlogsumexp_p",
    "cummax_p",
    "cummin_p",
    "cumprod_p",
    "cumsum_p",
    "device_put_p",
    "digamma_p",
    "div_p",
    "dot_general_p",
    "dynamic_slice_p",
    "dynamic_update_slice_p",
    "eq_p",
    "eq_to_p",
    "erf_inv_p",
    "erf_p",
    "erfc_p",
    "exp2_p",
    "exp_p",
    "expm1_p",
    "fft_p",
    "floor_p",
    "gather_p",
    "ge_p",
    "gt_p",
    "igamma_grad_a_p",
    "igamma_p",
    "igammac_p",
    "imag_p",
    "infeed_p",
    "integer_pow_p",
    "iota_p",
    "is_finite_p",
    "le_p",
    "lgamma_p",
    "linear_solve_p",
    "log1p_p",
    "log_p",
    "logistic_p",
    "lt_p",
    "max_p",
    "min_p",
    "mul_p",
    "ne_p",
    "neg_p",
    "nextafter_p",
    "not_p",
    "optimization_barrier_p",
    "or_p",
    "outfeed_p",
    "pad_p",
    "pmax_p",
    "pmin_p",
    "polygamma_p",
    "population_count_p",
    "pow_p",
    "ppermute_p",
    "psum_p",
    "random_gamma_grad_p",
    "real_p",
    "reduce_and_p",
    "reduce_max_p",
    "reduce_min_p",
    "reduce_or_p",
    "reduce_p",
    "reduce_precision_p",
    "reduce_prod_p",
    "reduce_sum_p",
    "reduce_window_max_p",
    "reduce_window_min_p",
    "reduce_window_p",
    "reduce_window_sum_p",
    "reduce_xor_p",
    "regularized_incomplete_beta_p",
    "rem_p",
    "reshape_p",
    "rev_p",
    "rng_bit_generator_p",
    "rng_uniform_p",
    "round_p",
    "rsqrt_p",
    "scan_p",
    "scatter_add_p",
    "scatter_apply_p",
    "scatter_max_p",
    "scatter_min_p",
    "scatter_mul_p",
    "scatter_p",
    "select_n_p",
    "select_p",
    "shift_left_p",
    "shift_right_arithmetic_p",
    "shift_right_logical_p",
    "sign_p",
    "sin_p",
    "sinh_p",
    "slice_p",
    "sort_p",
    "split_p",
    "sqrt_p",
    "square_p",
    "squeeze_p",
    "stop_gradient_p",
    "sub_p",
    "tan_p",
    "tanh_p",
    "top_k_p",
    "transpose_p",
    "while_p",
    "xor_p",
    "zeta_p",
]

__all__ = [
    "abs_p",
    "acos_p",
    "acosh_p",
    "add_p",
    "after_all_p",
    "all_gather_p",
    "all_to_all_p",
    "and_p",
    "approx_top_k_p",
    "argmax_p",
    "argmin_p",
    "asin_p",
    "asinh_p",
    "atan2_p",
    "atan_p",
    "atanh_p",
    "axis_index_p",
    "bessel_i0e_p",
    "bessel_i1e_p",
    "bitcast_convert_type_p",
    "broadcast_in_dim_p",
    "cbrt_p",
    "ceil_p",
    "clamp_p",
    "clz_p",
    "complex_p",
    "concatenate_p",
    "cond_p",
    "conj_p",
    "conv_general_dilated_p",
    "convert_element_type_p",
    "copy_p",
    "cos_p",
    "cosh_p",
    "create_token_p",
    "cumlogsumexp_p",
    "cummax_p",
    "cummin_p",
    "cumprod_p",
    "cumsum_p",
    "device_put_p",
    "digamma_p",
    "div_p",
    "dot_general_p",
    "dynamic_slice_p",
    "dynamic_update_slice_p",
    "eq_p",
    "eq_to_p",
    "erf_inv_p",
    "erf_p",
    "erfc_p",
    "exp2_p",
    "exp_p",
    "expm1_p",
    "fft_p",
    "floor_p",
    "gather_p",
    "ge_p",
    "gt_p",
    "igamma_grad_a_p",
    "igamma_p",
    "igammac_p",
    "imag_p",
    "infeed_p",
    "integer_pow_p",
    "iota_p",
    "is_finite_p",
    "le_p",
    "lgamma_p",
    "linear_solve_p",
    "log1p_p",
    "log_p",
    "logistic_p",
    "lt_p",
    "max_p",
    "min_p",
    "mul_p",
    "ne_p",
    "neg_p",
    "nextafter_p",
    "not_p",
    "optimization_barrier_p",
    "or_p",
    "outfeed_p",
    "pad_p",
    "pmax_p",
    "pmin_p",
    "polygamma_p",
    "population_count_p",
    "pow_p",
    "ppermute_p",
    "psum_p",
    "random_gamma_grad_p",
    "real_p",
    "reduce_and_p",
    "reduce_max_p",
    "reduce_min_p",
    "reduce_or_p",
    "reduce_p",
    "reduce_precision_p",
    "reduce_prod_p",
    "reduce_sum_p",
    "reduce_window_max_p",
    "reduce_window_min_p",
    "reduce_window_p",
    "reduce_window_sum_p",
    "reduce_xor_p",
    "regularized_incomplete_beta_p",
    "rem_p",
    "reshape_p",
    "rev_p",
    "rng_bit_generator_p",
    "rng_uniform_p",
    "round_p",
    "rsqrt_p",
    "scan_p",
    "scatter_add_p",
    "scatter_apply_p",
    "scatter_max_p",
    "scatter_min_p",
    "scatter_mul_p",
    "scatter_p",
    "select_n_p",
    "select_p",
    "shift_left_p",
    "shift_right_arithmetic_p",
    "shift_right_logical_p",
    "sign_p",
    "sin_p",
    "sinh_p",
    "slice_p",
    "sort_p",
    "split_p",
    "sqrt_p",
    "square_p",
    "squeeze_p",
    "stop_gradient_p",
    "sub_p",
    "tan_p",
    "tanh_p",
    "top_k_p",
    "transpose_p",
    "while_p",
    "xor_p",
    "zeta_p",
]
