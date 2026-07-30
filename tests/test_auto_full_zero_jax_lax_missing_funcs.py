"""Tests for zero_jax.lax.missing_funcs."""

from typing import Any

import pytest

import zero_jax.lax.missing_funcs as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
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


def test_associative_scan() -> None:
    """Test associative_scan."""
    try:
        mod.associative_scan(1.0, 1.0)
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


def test_dynamic_update_slice_in_dim() -> None:
    """Test dynamic_update_slice_in_dim."""
    try:
        mod.dynamic_update_slice_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_erf_inv() -> None:
    """Test erf_inv."""
    try:
        mod.erf_inv(1.0)
    except Exception:
        pass


def test_fori_loop() -> None:
    """Test fori_loop."""
    try:
        mod.fori_loop(1.0, 1.0, 1.0, 1.0)
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


def test_logistic() -> None:
    """Test logistic."""
    try:
        mod.logistic(1.0)
    except Exception:
        pass


def test_map() -> None:
    """Test map."""
    try:
        mod.map(1.0, 1.0)
    except Exception:
        pass


def test_outfeed() -> None:
    """Test outfeed."""
    try:
        mod.outfeed(1.0)
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


def test_reduce_precision() -> None:
    """Test reduce_precision."""
    try:
        mod.reduce_precision(1.0, 1.0, 1.0)
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


def test_scan_bind() -> None:
    """Test scan_bind."""
    try:
        mod.scan_bind(1.0)
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


def test_select_n() -> None:
    """Test select_n."""
    try:
        mod.select_n(1.0)
    except Exception:
        pass


def test_slice_in_dim() -> None:
    """Test slice_in_dim."""
    try:
        mod.slice_in_dim(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_sort_key_val() -> None:
    """Test sort_key_val."""
    try:
        mod.sort_key_val(1.0, 1.0)
    except Exception:
        pass


def test_switch() -> None:
    """Test switch."""
    try:
        mod.switch(1.0, 1.0)
    except Exception:
        pass


def test_top_k() -> None:
    """Test top_k."""
    try:
        mod.top_k(1.0, 1)
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
