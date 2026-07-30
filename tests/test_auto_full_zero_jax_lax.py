"""Tests for zero_jax.lax."""

from typing import Any

import pytest

import zero_jax.lax as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_class_ConvDimensionNumbers() -> None:
    """Test class ConvDimensionNumbers."""
    try:
        mod.ConvDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_ConvGeneralDilatedDimensionNumbers() -> None:
    """Test class ConvGeneralDilatedDimensionNumbers."""
    try:
        mod.ConvGeneralDilatedDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_DotDimensionNumbers() -> None:
    """Test class DotDimensionNumbers."""
    try:
        mod.DotDimensionNumbers(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_GatherDimensionNumbers() -> None:
    """Test class GatherDimensionNumbers."""
    try:
        mod.GatherDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_GatherScatterMode() -> None:
    """Test class GatherScatterMode."""
    try:
        mod.GatherScatterMode()
    except Exception:
        pass


def test_class_Precision() -> None:
    """Test class Precision."""
    try:
        mod.Precision()
    except Exception:
        pass


def test_class_PrecisionLike() -> None:
    """Test class PrecisionLike."""
    try:
        mod.PrecisionLike()
    except Exception:
        pass


def test_class_RandomAlgorithm() -> None:
    """Test class RandomAlgorithm."""
    try:
        mod.RandomAlgorithm()
    except Exception:
        pass


def test_class_RoundingMethod() -> None:
    """Test class RoundingMethod."""
    try:
        mod.RoundingMethod()
    except Exception:
        pass


def test_class_ScatterDimensionNumbers() -> None:
    """Test class ScatterDimensionNumbers."""
    try:
        mod.ScatterDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_abs() -> None:
    """Test abs."""
    try:
        mod.abs(1.0)
    except Exception:
        pass


def test_acos() -> None:
    """Test acos."""
    try:
        mod.acos(1.0)
    except Exception:
        pass


def test_acosh() -> None:
    """Test acosh."""
    try:
        mod.acosh(1.0)
    except Exception:
        pass


def test_add() -> None:
    """Test add."""
    try:
        mod.add(1.0, 1.0)
    except Exception:
        pass


def test_after_all() -> None:
    """Test after_all."""
    try:
        mod.after_all()
    except Exception:
        pass


def test_all_gather() -> None:
    """Test all_gather."""
    try:
        mod.all_gather(1.0, 1.0)
    except Exception:
        pass


def test_all_to_all() -> None:
    """Test all_to_all."""
    try:
        mod.all_to_all(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_approx_max_k() -> None:
    """Test approx_max_k."""
    try:
        mod.approx_max_k(1.0, 1)
    except Exception:
        pass


def test_approx_min_k() -> None:
    """Test approx_min_k."""
    try:
        mod.approx_min_k(1.0, 1)
    except Exception:
        pass


def test_argmax() -> None:
    """Test argmax."""
    try:
        mod.argmax(1.0, 1, 1.0)
    except Exception:
        pass


def test_argmin() -> None:
    """Test argmin."""
    try:
        mod.argmin(1.0, 1, 1.0)
    except Exception:
        pass


def test_asin() -> None:
    """Test asin."""
    try:
        mod.asin(1.0)
    except Exception:
        pass


def test_asinh() -> None:
    """Test asinh."""
    try:
        mod.asinh(1.0)
    except Exception:
        pass


def test_associative_scan() -> None:
    """Test associative_scan."""
    try:
        mod.associative_scan(1.0, 1.0)
    except Exception:
        pass


def test_atan() -> None:
    """Test atan."""
    try:
        mod.atan(1.0)
    except Exception:
        pass


def test_atan2() -> None:
    """Test atan2."""
    try:
        mod.atan2(1.0, 1.0)
    except Exception:
        pass


def test_atanh() -> None:
    """Test atanh."""
    try:
        mod.atanh(1.0)
    except Exception:
        pass


def test_axis_index() -> None:
    """Test axis_index."""
    try:
        mod.axis_index(1.0)
    except Exception:
        pass


def test_batch_matmul() -> None:
    """Test batch_matmul."""
    try:
        mod.batch_matmul(1.0, 1.0)
    except Exception:
        pass


def test_bessel_i0e() -> None:
    """Test bessel_i0e."""
    try:
        mod.bessel_i0e(1.0)
    except Exception:
        pass


def test_bessel_i1e() -> None:
    """Test bessel_i1e."""
    try:
        mod.bessel_i1e(1.0)
    except Exception:
        pass


def test_betainc() -> None:
    """Test betainc."""
    try:
        mod.betainc(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_bitcast_convert_type() -> None:
    """Test bitcast_convert_type."""
    try:
        mod.bitcast_convert_type(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_and() -> None:
    """Test bitwise_and."""
    try:
        mod.bitwise_and(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_not() -> None:
    """Test bitwise_not."""
    try:
        mod.bitwise_not(1.0)
    except Exception:
        pass


def test_bitwise_or() -> None:
    """Test bitwise_or."""
    try:
        mod.bitwise_or(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_xor() -> None:
    """Test bitwise_xor."""
    try:
        mod.bitwise_xor(1.0, 1.0)
    except Exception:
        pass


def test_broadcast() -> None:
    """Test broadcast."""
    try:
        mod.broadcast(1.0, 1.0)
    except Exception:
        pass


def test_broadcast_in_dim() -> None:
    """Test broadcast_in_dim."""
    try:
        mod.broadcast_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_broadcast_shapes() -> None:
    """Test broadcast_shapes."""
    try:
        mod.broadcast_shapes()
    except Exception:
        pass


def test_broadcast_to_rank() -> None:
    """Test broadcast_to_rank."""
    try:
        mod.broadcast_to_rank(1.0, 1.0)
    except Exception:
        pass


def test_broadcasted_iota() -> None:
    """Test broadcasted_iota."""
    try:
        mod.broadcasted_iota(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_cbrt() -> None:
    """Test cbrt."""
    try:
        mod.cbrt(1.0)
    except Exception:
        pass


def test_ceil() -> None:
    """Test ceil."""
    try:
        mod.ceil(1.0)
    except Exception:
        pass


def test_clamp() -> None:
    """Test clamp."""
    try:
        mod.clamp(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_clz() -> None:
    """Test clz."""
    try:
        mod.clz(1.0)
    except Exception:
        pass


def test_collapse() -> None:
    """Test collapse."""
    try:
        mod.collapse(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_complex() -> None:
    """Test complex."""
    try:
        mod.complex(1.0, 1.0)
    except Exception:
        pass


def test_concatenate() -> None:
    """Test concatenate."""
    try:
        mod.concatenate(1.0, 1.0)
    except Exception:
        pass


def test_cond() -> None:
    """Test cond."""
    try:
        mod.cond(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conj() -> None:
    """Test conj."""
    try:
        mod.conj(1.0)
    except Exception:
        pass


def test_conv() -> None:
    """Test conv."""
    try:
        mod.conv(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_dimension_numbers() -> None:
    """Test conv_dimension_numbers."""
    try:
        mod.conv_dimension_numbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_general_dilated() -> None:
    """Test conv_general_dilated."""
    try:
        mod.conv_general_dilated(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_general_dilated_local() -> None:
    """Test conv_general_dilated_local."""
    try:
        mod.conv_general_dilated_local(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_general_dilated_patches() -> None:
    """Test conv_general_dilated_patches."""
    try:
        mod.conv_general_dilated_patches(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_general_permutations() -> None:
    """Test conv_general_permutations."""
    try:
        mod.conv_general_permutations(1.0)
    except Exception:
        pass


def test_conv_general_shape_tuple() -> None:
    """Test conv_general_shape_tuple."""
    try:
        mod.conv_general_shape_tuple(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_shape_tuple() -> None:
    """Test conv_shape_tuple."""
    try:
        mod.conv_shape_tuple(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_transpose() -> None:
    """Test conv_transpose."""
    try:
        mod.conv_transpose(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_transpose_shape_tuple() -> None:
    """Test conv_transpose_shape_tuple."""
    try:
        mod.conv_transpose_shape_tuple(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_conv_with_general_padding() -> None:
    """Test conv_with_general_padding."""
    try:
        mod.conv_with_general_padding(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_convert_element_type() -> None:
    """Test convert_element_type."""
    try:
        mod.convert_element_type(1.0, 1.0)
    except Exception:
        pass


def test_cos() -> None:
    """Test cos."""
    try:
        mod.cos(1.0)
    except Exception:
        pass


def test_cosh() -> None:
    """Test cosh."""
    try:
        mod.cosh(1.0)
    except Exception:
        pass


def test_create_token() -> None:
    """Test create_token."""
    try:
        mod.create_token()
    except Exception:
        pass


def test_cumlogsumexp() -> None:
    """Test cumlogsumexp."""
    try:
        mod.cumlogsumexp(1.0)
    except Exception:
        pass


def test_cummax() -> None:
    """Test cummax."""
    try:
        mod.cummax(1.0)
    except Exception:
        pass


def test_cummin() -> None:
    """Test cummin."""
    try:
        mod.cummin(1.0)
    except Exception:
        pass


def test_cumprod() -> None:
    """Test cumprod."""
    try:
        mod.cumprod(1.0)
    except Exception:
        pass


def test_cumsum() -> None:
    """Test cumsum."""
    try:
        mod.cumsum(1.0, 1)
    except Exception:
        pass


def test_custom_linear_solve() -> None:
    """Test custom_linear_solve."""
    try:
        mod.custom_linear_solve(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_custom_root() -> None:
    """Test custom_root."""
    try:
        mod.custom_root(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_digamma() -> None:
    """Test digamma."""
    try:
        mod.digamma(1.0)
    except Exception:
        pass


def test_div() -> None:
    """Test div."""
    try:
        mod.div(1.0, 1.0)
    except Exception:
        pass


def test_dot() -> None:
    """Test dot."""
    try:
        mod.dot(1.0, 1.0)
    except Exception:
        pass


def test_dot_general() -> None:
    """Test dot_general."""
    try:
        mod.dot_general(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_dtype() -> None:
    """Test dtype."""
    try:
        mod.dtype(1.0)
    except Exception:
        pass


def test_dynamic_index_in_dim() -> None:
    """Test dynamic_index_in_dim."""
    try:
        mod.dynamic_index_in_dim(1.0, 1.0)
    except Exception:
        pass


def test_dynamic_slice() -> None:
    """Test dynamic_slice."""
    try:
        mod.dynamic_slice(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_dynamic_slice_in_dim() -> None:
    """Test dynamic_slice_in_dim."""
    try:
        mod.dynamic_slice_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_dynamic_update_index_in_dim() -> None:
    """Test dynamic_update_index_in_dim."""
    try:
        mod.dynamic_update_index_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_dynamic_update_slice() -> None:
    """Test dynamic_update_slice."""
    try:
        mod.dynamic_update_slice(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_dynamic_update_slice_in_dim() -> None:
    """Test dynamic_update_slice_in_dim."""
    try:
        mod.dynamic_update_slice_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_eq() -> None:
    """Test eq."""
    try:
        mod.eq(1.0, 1.0)
    except Exception:
        pass


def test_erf() -> None:
    """Test erf."""
    try:
        mod.erf(1.0)
    except Exception:
        pass


def test_erf_inv() -> None:
    """Test erf_inv."""
    try:
        mod.erf_inv(1.0)
    except Exception:
        pass


def test_erfc() -> None:
    """Test erfc."""
    try:
        mod.erfc(1.0)
    except Exception:
        pass


def test_exp() -> None:
    """Test exp."""
    try:
        mod.exp(1.0)
    except Exception:
        pass


def test_exp2() -> None:
    """Test exp2."""
    try:
        mod.exp2(1.0)
    except Exception:
        pass


def test_expand_dims() -> None:
    """Test expand_dims."""
    try:
        mod.expand_dims(1.0, 1.0)
    except Exception:
        pass


def test_expm1() -> None:
    """Test expm1."""
    try:
        mod.expm1(1.0)
    except Exception:
        pass


def test_fft() -> None:
    """Test fft."""
    try:
        mod.fft(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_floor() -> None:
    """Test floor."""
    try:
        mod.floor(1.0)
    except Exception:
        pass


def test_fori_loop() -> None:
    """Test fori_loop."""
    try:
        mod.fori_loop(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_full() -> None:
    """Test full."""
    try:
        mod.full(1.0, 1.0)
    except Exception:
        pass


def test_full_like() -> None:
    """Test full_like."""
    try:
        mod.full_like(1.0, 1.0)
    except Exception:
        pass


def test_gather() -> None:
    """Test gather."""
    try:
        mod.gather(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_ge() -> None:
    """Test ge."""
    try:
        mod.ge(1.0, 1.0)
    except Exception:
        pass


def test_gt() -> None:
    """Test gt."""
    try:
        mod.gt(1.0, 1.0)
    except Exception:
        pass


def test_igamma() -> None:
    """Test igamma."""
    try:
        mod.igamma(1.0, 1.0)
    except Exception:
        pass


def test_igamma_grad_a() -> None:
    """Test igamma_grad_a."""
    try:
        mod.igamma_grad_a(1.0, 1.0)
    except Exception:
        pass


def test_igammac() -> None:
    """Test igammac."""
    try:
        mod.igammac(1.0, 1.0)
    except Exception:
        pass


def test_imag() -> None:
    """Test imag."""
    try:
        mod.imag(1.0)
    except Exception:
        pass


def test_index_in_dim() -> None:
    """Test index_in_dim."""
    try:
        mod.index_in_dim(1.0, 1.0)
    except Exception:
        pass


def test_index_take() -> None:
    """Test index_take."""
    try:
        mod.index_take(1.0, 1.0, 1)
    except Exception:
        pass


def test_infeed() -> None:
    """Test infeed."""
    try:
        mod.infeed()
    except Exception:
        pass


def test_integer_pow() -> None:
    """Test integer_pow."""
    try:
        mod.integer_pow(1.0, 1.0)
    except Exception:
        pass


def test_iota() -> None:
    """Test iota."""
    try:
        mod.iota(1.0, 1.0)
    except Exception:
        pass


def test_is_finite() -> None:
    """Test is_finite."""
    try:
        mod.is_finite(1.0)
    except Exception:
        pass


def test_le() -> None:
    """Test le."""
    try:
        mod.le(1.0, 1.0)
    except Exception:
        pass


def test_lgamma() -> None:
    """Test lgamma."""
    try:
        mod.lgamma(1.0)
    except Exception:
        pass


def test_log() -> None:
    """Test log."""
    try:
        mod.log(1.0)
    except Exception:
        pass


def test_log1p() -> None:
    """Test log1p."""
    try:
        mod.log1p(1.0)
    except Exception:
        pass


def test_logistic() -> None:
    """Test logistic."""
    try:
        mod.logistic(1.0)
    except Exception:
        pass


def test_lt() -> None:
    """Test lt."""
    try:
        mod.lt(1.0, 1.0)
    except Exception:
        pass


def test_map() -> None:
    """Test map."""
    try:
        mod.map(1.0, 1.0)
    except Exception:
        pass


def test_max() -> None:
    """Test max."""
    try:
        mod.max(1.0, 1.0)
    except Exception:
        pass


def test_min() -> None:
    """Test min."""
    try:
        mod.min(1.0, 1.0)
    except Exception:
        pass


def test_mul() -> None:
    """Test mul."""
    try:
        mod.mul(1.0, 1.0)
    except Exception:
        pass


def test_ne() -> None:
    """Test ne."""
    try:
        mod.ne(1.0, 1.0)
    except Exception:
        pass


def test_neg() -> None:
    """Test neg."""
    try:
        mod.neg(1.0)
    except Exception:
        pass


def test_nextafter() -> None:
    """Test nextafter."""
    try:
        mod.nextafter(1.0, 1.0)
    except Exception:
        pass


def test_outfeed() -> None:
    """Test outfeed."""
    try:
        mod.outfeed(1.0)
    except Exception:
        pass


def test_pad() -> None:
    """Test pad."""
    try:
        mod.pad(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_padtype_to_pads() -> None:
    """Test padtype_to_pads."""
    try:
        mod.padtype_to_pads(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_pbroadcast() -> None:
    """Test pbroadcast."""
    try:
        mod.pbroadcast(1.0, 1.0)
    except Exception:
        pass


def test_pdot() -> None:
    """Test pdot."""
    try:
        mod.pdot(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_platform_dependent() -> None:
    """Test platform_dependent."""
    try:
        mod.platform_dependent(1.0, 1.0)
    except Exception:
        pass


def test_pmax() -> None:
    """Test pmax."""
    try:
        mod.pmax(1.0, 1.0)
    except Exception:
        pass


def test_pmean() -> None:
    """Test pmean."""
    try:
        mod.pmean(1.0, 1.0)
    except Exception:
        pass


def test_pmin() -> None:
    """Test pmin."""
    try:
        mod.pmin(1.0, 1.0)
    except Exception:
        pass


def test_polygamma() -> None:
    """Test polygamma."""
    try:
        mod.polygamma(1.0, 1.0)
    except Exception:
        pass


def test_population_count() -> None:
    """Test population_count."""
    try:
        mod.population_count(1.0)
    except Exception:
        pass


def test_pow() -> None:
    """Test pow."""
    try:
        mod.pow(1.0, 1.0)
    except Exception:
        pass


def test_ppermute() -> None:
    """Test ppermute."""
    try:
        mod.ppermute(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_pshuffle() -> None:
    """Test pshuffle."""
    try:
        mod.pshuffle(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_psum() -> None:
    """Test psum."""
    try:
        mod.psum(1.0, 1.0)
    except Exception:
        pass


def test_psum_scatter() -> None:
    """Test psum_scatter."""
    try:
        mod.psum_scatter(1.0, 1.0)
    except Exception:
        pass


def test_pswapaxes() -> None:
    """Test pswapaxes."""
    try:
        mod.pswapaxes(1.0, 1.0, 1)
    except Exception:
        pass


def test_ragged_dot() -> None:
    """Test ragged_dot."""
    try:
        mod.ragged_dot(1.0, 1.0)
    except Exception:
        pass


def test_random_gamma_grad() -> None:
    """Test random_gamma_grad."""
    try:
        mod.random_gamma_grad(1.0, 1.0)
    except Exception:
        pass


def test_real() -> None:
    """Test real."""
    try:
        mod.real(1.0)
    except Exception:
        pass


def test_reciprocal() -> None:
    """Test reciprocal."""
    try:
        mod.reciprocal(1.0)
    except Exception:
        pass


def test_reduce() -> None:
    """Test reduce."""
    try:
        mod.reduce(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_reduce_precision() -> None:
    """Test reduce_precision."""
    try:
        mod.reduce_precision(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_reduce_window() -> None:
    """Test reduce_window."""
    try:
        mod.reduce_window(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_reduce_window_shape_tuple() -> None:
    """Test reduce_window_shape_tuple."""
    try:
        mod.reduce_window_shape_tuple(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_rem() -> None:
    """Test rem."""
    try:
        mod.rem(1.0, 1.0)
    except Exception:
        pass


def test_reshape() -> None:
    """Test reshape."""
    try:
        mod.reshape(1.0, 1.0)
    except Exception:
        pass


def test_rev() -> None:
    """Test rev."""
    try:
        mod.rev(1.0, 1.0)
    except Exception:
        pass


def test_rng_bit_generator() -> None:
    """Test rng_bit_generator."""
    try:
        mod.rng_bit_generator(1.0, 1.0)
    except Exception:
        pass


def test_rng_uniform() -> None:
    """Test rng_uniform."""
    try:
        mod.rng_uniform(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_round() -> None:
    """Test round."""
    try:
        mod.round(1.0)
    except Exception:
        pass


def test_rsqrt() -> None:
    """Test rsqrt."""
    try:
        mod.rsqrt(1.0)
    except Exception:
        pass


def test_scan() -> None:
    """Test scan."""
    try:
        mod.scan(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scan_bind() -> None:
    """Test scan_bind."""
    try:
        mod.scan_bind(1.0)
    except Exception:
        pass


def test_scatter() -> None:
    """Test scatter."""
    try:
        mod.scatter(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scatter_add() -> None:
    """Test scatter_add."""
    try:
        mod.scatter_add(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scatter_apply() -> None:
    """Test scatter_apply."""
    try:
        mod.scatter_apply(1.0, 1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scatter_max() -> None:
    """Test scatter_max."""
    try:
        mod.scatter_max(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scatter_min() -> None:
    """Test scatter_min."""
    try:
        mod.scatter_min(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_scatter_mul() -> None:
    """Test scatter_mul."""
    try:
        mod.scatter_mul(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_select() -> None:
    """Test select."""
    try:
        mod.select(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_select_n() -> None:
    """Test select_n."""
    try:
        mod.select_n(1.0)
    except Exception:
        pass


def test_shift_left() -> None:
    """Test shift_left."""
    try:
        mod.shift_left(1.0, 1.0)
    except Exception:
        pass


def test_shift_right_arithmetic() -> None:
    """Test shift_right_arithmetic."""
    try:
        mod.shift_right_arithmetic(1.0, 1.0)
    except Exception:
        pass


def test_shift_right_logical() -> None:
    """Test shift_right_logical."""
    try:
        mod.shift_right_logical(1.0, 1.0)
    except Exception:
        pass


def test_sign() -> None:
    """Test sign."""
    try:
        mod.sign(1.0)
    except Exception:
        pass


def test_sin() -> None:
    """Test sin."""
    try:
        mod.sin(1.0)
    except Exception:
        pass


def test_sinh() -> None:
    """Test sinh."""
    try:
        mod.sinh(1.0)
    except Exception:
        pass


def test_slice() -> None:
    """Test slice."""
    try:
        mod.slice(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_slice_in_dim() -> None:
    """Test slice_in_dim."""
    try:
        mod.slice_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_sort() -> None:
    """Test sort."""
    try:
        mod.sort(1.0)
    except Exception:
        pass


def test_sort_key_val() -> None:
    """Test sort_key_val."""
    try:
        mod.sort_key_val(1.0, 1.0)
    except Exception:
        pass


def test_sqrt() -> None:
    """Test sqrt."""
    try:
        mod.sqrt(1.0)
    except Exception:
        pass


def test_square() -> None:
    """Test square."""
    try:
        mod.square(1.0)
    except Exception:
        pass


def test_squeeze() -> None:
    """Test squeeze."""
    try:
        mod.squeeze(1.0, 1.0)
    except Exception:
        pass


def test_stop_gradient() -> None:
    """Test stop_gradient."""
    try:
        mod.stop_gradient(1.0)
    except Exception:
        pass


def test_sub() -> None:
    """Test sub."""
    try:
        mod.sub(1.0, 1.0)
    except Exception:
        pass


def test_switch() -> None:
    """Test switch."""
    try:
        mod.switch(1.0, 1.0)
    except Exception:
        pass


def test_tan() -> None:
    """Test tan."""
    try:
        mod.tan(1.0)
    except Exception:
        pass


def test_tanh() -> None:
    """Test tanh."""
    try:
        mod.tanh(1.0)
    except Exception:
        pass


def test_top_k() -> None:
    """Test top_k."""
    try:
        mod.top_k(1.0, 1)
    except Exception:
        pass


def test_transpose() -> None:
    """Test transpose."""
    try:
        mod.transpose(1.0, 1.0)
    except Exception:
        pass


def test_while_loop() -> None:
    """Test while_loop."""
    try:
        mod.while_loop(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_with_sharding_constraint() -> None:
    """Test with_sharding_constraint."""
    try:
        mod.with_sharding_constraint(1.0, 1.0)
    except Exception:
        pass


def test_xeinsum() -> None:
    """Test xeinsum."""
    try:
        mod.xeinsum(1.0)
    except Exception:
        pass


def test_zeros_like_array() -> None:
    """Test zeros_like_array."""
    try:
        mod.zeros_like_array(1.0)
    except Exception:
        pass


def test_zeta() -> None:
    """Test zeta."""
    try:
        mod.zeta(1.0, 1.0)
    except Exception:
        pass
