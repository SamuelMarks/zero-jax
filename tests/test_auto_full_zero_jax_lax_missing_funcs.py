"""Tests for zero_jax.lax.missing_funcs."""

from __future__ import annotations

import pytest

import zero_jax.lax.missing_funcs as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_after_all() -> None:
    """Test after_all."""
    token = mod.after_all()
    assert token.shape == ()


def test_all_gather() -> None:
    """Test all_gather."""
    res = mod.all_gather(jnp.array([1.0, 2.0]), 0)
    assert res.shape == (1, 2)
    assert_allclose(res, [[1.0, 2.0]])


def test_all_to_all() -> None:
    """Test all_to_all."""
    res = mod.all_to_all(jnp.array([0.5, 1.5]), "i", 0, 0)
    assert res.shape == (2,)
    assert_allclose(res, [0.5, 1.5])


def test_approx_max_k() -> None:
    """Test approx_max_k."""
    vals, idx = mod.approx_max_k(jnp.array([1.0, 4.0, 2.0]), 2)
    assert vals.shape == (2,)
    assert idx.shape == (2,)
    assert_allclose(vals, [4.0, 2.0])


def test_approx_min_k() -> None:
    """Test approx_min_k."""
    vals, idx = mod.approx_min_k(jnp.array([1.0, 4.0, 2.0]), 2)
    assert vals.shape == (2,)
    assert idx.shape == (2,)
    assert_allclose(vals, [1.0, 2.0])


def test_associative_scan() -> None:
    """Test associative_scan."""
    res = mod.associative_scan(lambda a, b: a + b, jnp.array([1.0, 2.0, 3.0]))
    assert res.shape == (3,)
    assert_allclose(res, [1.0, 3.0, 6.0])


def test_axis_index() -> None:
    """Test axis_index."""
    with pytest.raises(TypeError):
        mod.axis_index("i")


def test_batch_matmul() -> None:
    """Test batch_matmul."""
    res = mod.batch_matmul(jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]))
    assert_allclose(res, 2.5)


def test_bessel_i0e() -> None:
    """Test bessel_i0e."""
    res = mod.bessel_i0e(jnp.array([0.0, 1.0]))
    assert res.shape == (2,)
    assert_allclose(res[0], 1.0, atol=1e-4)


def test_bessel_i1e() -> None:
    """Test bessel_i1e."""
    res = mod.bessel_i1e(jnp.array([0.0, 1.0]))
    assert res.shape == (2,)
    assert_allclose(res[0], 0.0, atol=1e-4)


def test_betainc() -> None:
    """Test betainc."""
    res = mod.betainc(jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]), jnp.array([0.25, 0.75]))
    assert res.shape == (2,)
    assert_allclose(res, [0.25, 0.75], atol=1e-4)


def test_bitcast_convert_type() -> None:
    """Test bitcast_convert_type."""
    res = mod.bitcast_convert_type(jnp.array([0.5, 1.5], dtype=jnp.float32), jnp.float32)
    assert res.shape == (2,)


def test_broadcast_to_rank() -> None:
    """Test broadcast_to_rank."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.broadcast_to_rank(jnp.array([1.0, 2.0]), 2)


def test_broadcasted_iota() -> None:
    """Test broadcasted_iota."""
    res = mod.broadcasted_iota(3, (2, 3), 1)
    assert res.shape == (2, 3)


def test_clz() -> None:
    """Test clz."""
    res = mod.clz(jnp.array([1, 2], dtype=jnp.int32))
    assert res.shape == (2,)
    assert_allclose(res, [31, 30])


def test_collapse() -> None:
    """Test collapse."""
    res = mod.collapse(jnp.ones((2, 3, 4)), 0, 2)
    assert res.shape == (6, 4)


def test_complex() -> None:
    """Test complex."""
    res = mod.complex(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert res.shape == (2,)
    assert_allclose(jnp.real(res), [1.0, 2.0])


def test_conv() -> None:
    """Test conv."""
    res = mod.conv(jnp.array([1.0, 2.0]), jnp.array([1.0, 1.0]), (1,), "SAME")
    assert res.shape == (2,)
    assert_allclose(res, [1.0, 3.0])
    res_3d = mod.conv(jnp.ones((1, 1, 4)), jnp.ones((1, 1, 2)), (1,), "VALID")
    assert res_3d.shape == (1, 1, 3)


def test_conv_dimension_numbers() -> None:
    """Test conv_dimension_numbers."""
    res = mod.conv_dimension_numbers((1, 1, 4), (1, 1, 2), None)
    assert res.lhs_spec == (0, 1, 2)
    res_existing = mod.conv_dimension_numbers((1, 1, 4), (1, 1, 2), res)
    assert res_existing is res


def test_conv_general_dilated_local() -> None:
    """Test conv_general_dilated_local."""
    res = mod.conv_general_dilated_local(jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]))
    assert res.shape == (1,)


def test_conv_general_dilated_patches() -> None:
    """Test conv_general_dilated_patches."""
    res = mod.conv_general_dilated_patches(jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]))
    assert res.shape == (1,)


def test_conv_general_permutations() -> None:
    """Test conv_general_permutations."""
    res = mod.conv_general_permutations(jnp.array([0, 1, 2]))
    assert res == ((0, 1, 2), (0, 1, 2), (0, 1, 2))
    cdn = mod.conv_dimension_numbers((1, 1, 4), (1, 1, 2), None)
    assert mod.conv_general_permutations(cdn) == ((0, 1, 2), (0, 1, 2), (0, 1, 2))


def test_conv_general_shape_tuple() -> None:
    """Test conv_general_shape_tuple."""
    res = mod.conv_general_shape_tuple((1, 2, 4), (3, 2, 2), (1,), "VALID")
    assert res == (1, 3, 3)
    res_short = mod.conv_general_shape_tuple((2, 4), (2, 2), (1,), "VALID")
    assert res_short == (2, 4)


def test_conv_shape_tuple() -> None:
    """Test conv_shape_tuple."""
    res = mod.conv_shape_tuple((1, 2, 4), (3, 2, 2), (1,), "SAME")
    assert res == (1, 3, 4)


def test_conv_transpose() -> None:
    """Test conv_transpose."""
    lhs = jnp.ones((1, 1, 4))
    rhs = jnp.ones((1, 1, 2))
    res = mod.conv_transpose(lhs, rhs, (1,), "VALID")
    assert res.shape == (1, 1, 5)


def test_conv_transpose_shape_tuple() -> None:
    """Test conv_transpose_shape_tuple."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.conv_transpose_shape_tuple((1, 4, 1), (2, 1, 1), (1,), "VALID")


def test_conv_with_general_padding() -> None:
    """Test conv_with_general_padding."""
    res = mod.conv_with_general_padding(jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]), jnp.array([0.5, 1.5]))
    assert res.shape == (1,)


def test_convert_element_type() -> None:
    """Test convert_element_type."""
    res = mod.convert_element_type(jnp.array([1.0, 2.0]), "int32")
    assert res.shape == (2,)
    assert_allclose(res, [1, 2])


def test_create_token() -> None:
    """Test create_token."""
    res = mod.create_token()
    assert res.shape == ()


def test_cumlogsumexp() -> None:
    """Test cumlogsumexp."""
    res = mod.cumlogsumexp(jnp.array([0.0, 0.0]))
    assert res.shape == (2,)
    assert_allclose(res[0], 0.0, atol=1e-4)


def test_cummax() -> None:
    """Test cummax."""
    res = mod.cummax(jnp.array([2.0, 1.0, 3.0]))
    assert_allclose(res, [2.0, 2.0, 3.0])


def test_cummin() -> None:
    """Test cummin."""
    res = mod.cummin(jnp.array([2.0, 1.0, 3.0]))
    assert_allclose(res, [2.0, 1.0, 1.0])


def test_cumprod() -> None:
    """Test cumprod."""
    res = mod.cumprod(jnp.array([2.0, 3.0]))
    assert_allclose(res, [2.0, 6.0])


def test_custom_linear_solve() -> None:
    """Test custom_linear_solve."""
    res = mod.custom_linear_solve(jnp.eye(2), jnp.eye(2), lambda m, b: b)
    assert res.shape == (2, 2)


def test_custom_root() -> None:
    """Test custom_root."""
    with pytest.raises(TypeError):
        mod.custom_root(lambda x: x - 2.0, jnp.array(1.0), lambda f, x: jnp.array(2.0), lambda g, y: y)


def test_dtype() -> None:
    """Test dtype."""
    res = mod.dtype(jnp.array([0.5, 1.5]))
    assert "float" in str(res).lower()
    assert mod.dtype(5) is None


def test_dynamic_index_in_dim() -> None:
    """Test dynamic_index_in_dim."""
    res = mod.dynamic_index_in_dim(jnp.array([10.0, 20.0, 30.0]), 1, 0, False)
    assert_allclose(res, 20.0)


def test_dynamic_slice_in_dim() -> None:
    """Test dynamic_slice_in_dim."""
    res = mod.dynamic_slice_in_dim(jnp.array([10.0, 20.0, 30.0]), 1, 2)
    assert_allclose(res, [20.0, 30.0])


def test_dynamic_update_index_in_dim() -> None:
    """Test dynamic_update_index_in_dim."""
    res = mod.dynamic_update_index_in_dim(jnp.array([10.0, 20.0]), jnp.array(99.0), 1, 0)
    assert_allclose(res, [10.0, 99.0])


def test_dynamic_update_slice_in_dim() -> None:
    """Test dynamic_update_slice_in_dim."""
    res = mod.dynamic_update_slice_in_dim(jnp.array([10.0, 20.0, 30.0]), jnp.array([99.0]), 1, 0)
    assert_allclose(res, [10.0, 99.0, 30.0])


def test_erf_inv() -> None:
    """Test erf_inv."""
    res = mod.erf_inv(jnp.array([0.0, 0.5]))
    assert res.shape == (2,)
    assert_allclose(res[0], 0.0, atol=1e-4)


def test_fori_loop() -> None:
    """Test fori_loop."""
    res = mod.fori_loop(0, 3, lambda i, v: v + 1.0, jnp.array(0.0))
    assert res.item() == 3.0


def test_igamma() -> None:
    """Test igamma."""
    res = mod.igamma(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_igamma_grad_a() -> None:
    """Test igamma_grad_a."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.igamma_grad_a(jnp.array([1.0]), jnp.array([1.0]))


def test_igammac() -> None:
    """Test igammac."""
    res = mod.igammac(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_index_in_dim() -> None:
    """Test index_in_dim."""
    res = mod.index_in_dim(jnp.array([10.0, 20.0, 30.0]), 1, 0, False)
    assert res.item() == 20.0


def test_index_take() -> None:
    """Test index_take."""
    res = mod.index_take(jnp.array([10.0, 20.0, 30.0]), jnp.array([2, 0]), jnp.array([0]))
    assert_allclose(res, [30.0, 10.0])


def test_infeed() -> None:
    """Test infeed."""
    res = mod.infeed()
    assert res.shape == ()


def test_integer_pow() -> None:
    """Test integer_pow."""
    res = mod.integer_pow(jnp.array([2.0, 3.0]), 2)
    assert_allclose(res, [4.0, 9.0])


def test_iota() -> None:
    """Test iota."""
    res = mod.iota(jnp.float32, 4)
    assert res.shape == (4,)
    assert_allclose(res, [0.0, 1.0, 2.0, 3.0])


def test_is_finite() -> None:
    """Test is_finite."""
    res = mod.is_finite(jnp.array([0.5, 1.5]))
    assert list(res.__array__()) == [True, True]


def test_logistic() -> None:
    """Test logistic."""
    res = mod.logistic(jnp.array([0.0]))
    assert_allclose(res, [0.5])


def test_map() -> None:
    """Test map."""
    res = mod.map(lambda x: x * 2.0, jnp.array([1.0, 2.0, 3.0]))
    assert res.shape == (3,)
    assert_allclose(res, [2.0, 4.0, 6.0])


def test_outfeed() -> None:
    """Test outfeed."""
    mod.outfeed(jnp.array([1.0]))
    assert callable(mod.outfeed)


def test_padtype_to_pads() -> None:
    """Test padtype_to_pads."""
    assert mod.padtype_to_pads((4, 4), (2, 2), (1, 1), "VALID") == [(0, 0), (0, 0)]
    assert mod.padtype_to_pads((4, 4), (3, 3), (1, 1), "SAME") == [(1, 1), (1, 1)]
    assert mod.padtype_to_pads((4, 4), (2, 2), (1, 1), jnp.array([[1, 2], [0, 1]])) == [(1, 2), (0, 1)]
    assert mod.padtype_to_pads((4, 4), (2, 2), (1, 1), jnp.array([1, 2])) == [(1, 1), (2, 2)]


def test_pbroadcast() -> None:
    """Test pbroadcast."""
    res = mod.pbroadcast(jnp.array([1.0, 2.0]), "i")
    assert_allclose(res, [1.0, 2.0])


def test_pdot() -> None:
    """Test pdot."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.pdot(jnp.array([1.0]), jnp.array([2.0]), "i")


def test_platform_dependent() -> None:
    """Test platform_dependent."""
    res = mod.platform_dependent(jnp.array([0.5, 1.5]), jnp.array([2.5, 3.5]))
    assert_allclose(res, [0.5, 1.5])


def test_pmax() -> None:
    """Test pmax."""
    res = mod.pmax(jnp.array([1.0, 2.0]), "i")
    assert_allclose(res, [1.0, 2.0])


def test_pmin() -> None:
    """Test pmin."""
    res = mod.pmin(jnp.array([1.0, 2.0]), "i")
    assert_allclose(res, [1.0, 2.0])


def test_polygamma() -> None:
    """Test polygamma."""
    res = mod.polygamma(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5]))
    assert res.shape == (2,)


def test_population_count() -> None:
    """Test population_count."""
    res = mod.population_count(jnp.array([1, 3], dtype=jnp.int32))
    assert res.shape == (2,)
    assert_allclose(res, [1, 2])


def test_pow() -> None:
    """Test pow."""
    res = mod.pow(jnp.array([2.0, 3.0]), jnp.array([3.0, 2.0]))
    assert_allclose(res, [8.0, 9.0])


def test_ppermute() -> None:
    """Test ppermute."""
    res = mod.ppermute(jnp.array([1.0, 2.0]), "i", [(0, 0)])
    assert_allclose(res, [1.0, 2.0])


def test_pshuffle() -> None:
    """Test pshuffle."""
    res = mod.pshuffle(jnp.array([1.0, 2.0]), "i", [0])
    assert_allclose(res, [1.0, 2.0])


def test_psum_scatter() -> None:
    """Test psum_scatter."""
    res = mod.psum_scatter(jnp.array([0.5, 1.5]), "i")
    assert res.shape == (2,)


def test_pswapaxes() -> None:
    """Test pswapaxes."""
    res = mod.pswapaxes(jnp.array([1.0, 2.0]), "i", 0)
    assert_allclose(res, [1.0, 2.0])


def test_ragged_dot() -> None:
    """Test ragged_dot."""
    res = mod.ragged_dot(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert_allclose(res, 11.0)


def test_random_gamma_grad() -> None:
    """Test random_gamma_grad."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.random_gamma_grad(jnp.array([1.0]), jnp.array([1.0]))


def test_reduce_precision() -> None:
    """Test reduce_precision."""
    res = mod.reduce_precision(jnp.array([0.5, 1.5]), 5, 10)
    assert res.shape == (2,)


def test_reduce_window_shape_tuple() -> None:
    """Test reduce_window_shape_tuple."""
    res = mod.reduce_window_shape_tuple((4, 4), (2, 2), (2, 2), "VALID")
    assert res == (2, 2)


def test_rem() -> None:
    """Test rem."""
    res = mod.rem(jnp.array([5.0, 7.0]), jnp.array([2.0, 3.0]))
    assert_allclose(res, [1.0, 1.0])


def test_rev() -> None:
    """Test rev."""
    res = mod.rev(jnp.array([1.0, 2.0, 3.0]), jnp.array([0]))
    assert_allclose(res, [3.0, 2.0, 1.0])
    scalar_res = mod.rev(jnp.array(5.0), jnp.array([]))
    assert_allclose(scalar_res, 5.0)


def test_rng_bit_generator() -> None:
    """Test rng_bit_generator."""
    new_key, bits = mod.rng_bit_generator(jnp.array([0, 1]), (2, 3))
    assert new_key.shape == (2,)
    assert bits.shape == (2, 3)


def test_rng_uniform() -> None:
    """Test rng_uniform."""
    res = mod.rng_uniform(jnp.array(1.0), jnp.array(2.0), (2, 3))
    assert res.shape == (2, 3)
    assert bool(jnp.all((res >= 1.0) & (res <= 2.0)))


def test_scan_bind() -> None:
    """Test scan_bind."""
    assert_allclose(mod.scan_bind(jnp.array([1.0])), [1.0])
    assert mod.scan_bind(lambda x: x + 1, 2) == 3


def test_scatter_apply() -> None:
    """Test scatter_apply."""
    with pytest.raises(RuntimeError):
        mod.scatter_apply(jnp.array([1.0, 2.0]), jnp.array([0.0]), lambda x: x, (2,), (1,))


def test_scatter_max() -> None:
    """Test scatter_max."""
    with pytest.raises(TypeError):
        mod.scatter_max(jnp.array([1.0, 2.0]), jnp.array([0]), jnp.array([9.0]), (0,))


def test_scatter_min() -> None:
    """Test scatter_min."""
    with pytest.raises(TypeError):
        mod.scatter_min(jnp.array([1.0, 2.0]), jnp.array([0]), jnp.array([9.0]), (0,))


def test_scatter_mul() -> None:
    """Test scatter_mul."""
    with pytest.raises(TypeError):
        mod.scatter_mul(jnp.array([1.0, 2.0]), jnp.array([0]), jnp.array([9.0]), (0,))


def test_select_n() -> None:
    """Test select_n."""
    res = mod.select_n(jnp.array([0, 1], dtype=jnp.int32), jnp.array([10.0, 20.0]), jnp.array([30.0, 40.0]))
    assert res.shape == (2,)
    assert_allclose(res, [10.0, 40.0])


def test_slice_in_dim() -> None:
    """Test slice_in_dim."""
    res = mod.slice_in_dim(jnp.array([10.0, 20.0, 30.0]), 0, 2)
    assert res.shape == (2,)
    assert_allclose(res, [10.0, 20.0])


def test_sort_key_val() -> None:
    """Test sort_key_val."""
    keys, vals = mod.sort_key_val(jnp.array([2.0, 1.0]), jnp.array([20.0, 10.0]))
    assert_allclose(keys, [1.0, 2.0])
    assert_allclose(vals, [10.0, 20.0])


def test_switch() -> None:
    """Test switch."""
    res = mod.switch(1, [lambda x: x + 1.0, lambda x: x + 2.0], jnp.array(3.0))
    assert res.item() == 4.0


def test_top_k() -> None:
    """Test top_k."""
    vals, idx = mod.top_k(jnp.array([1.0, 4.0, 2.0]), 2)
    assert vals.shape == (2,)
    assert idx.shape == (2,)
    assert_allclose(vals, [4.0, 2.0])


def test_while_loop() -> None:
    """Test while_loop."""
    res = mod.while_loop(lambda v: v < 3.0, lambda v: v + 1.0, jnp.array(0.0))
    assert res.item() == 3.0


def test_with_sharding_constraint() -> None:
    """Test with_sharding_constraint."""
    res = mod.with_sharding_constraint(jnp.array([0.5, 1.5]), None)
    assert_allclose(res, [0.5, 1.5])


def test_xeinsum() -> None:
    """Test xeinsum."""
    res = mod.xeinsum("i,i->", jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert_allclose(res, 11.0)


def test_zeros_like_array() -> None:
    """Test zeros_like_array."""
    res = mod.zeros_like_array(jnp.array([0.5, 1.5]))
    assert_allclose(res, [0.0, 0.0])


def test_zeta() -> None:
    """Test zeta."""
    res = mod.zeta(jnp.array([2.0]), jnp.array([1.0]))
    assert res.shape == (1,)
