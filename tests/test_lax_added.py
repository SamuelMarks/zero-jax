import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import zero_jax.lax as jlax
import zero_jax.numpy as jnp


def test_lax_types():
    assert isinstance(
        jlax.ConvDimensionNumbers(None, None, None), jlax.ConvDimensionNumbers
    )
    assert isinstance(
        jlax.ConvGeneralDilatedDimensionNumbers(None, None, None),
        jlax.ConvGeneralDilatedDimensionNumbers,
    )
    assert isinstance(
        jlax.DotDimensionNumbers(None, None, None, None), jlax.DotDimensionNumbers
    )
    assert isinstance(
        jlax.GatherDimensionNumbers(None, None, None), jlax.GatherDimensionNumbers
    )
    assert isinstance(
        jlax.ScatterDimensionNumbers(None, None, None), jlax.ScatterDimensionNumbers
    )
    assert jlax.Precision.DEFAULT == "DEFAULT"
    assert jlax.abs_p.name == "abs_p"


def test_lax_funcs():
    a = jnp.array([1, 2, 3])
    b = jnp.array([2, 3, 4])

    assert jlax.approx_max_k(a, 1)[0].shape is not None
    assert jlax.approx_min_k(a, 1)[0].shape is not None
    assert jlax.bitcast_convert_type(a, np.float32).shape is not None
    assert jlax.broadcast_to_rank(a, 2).shape is not None

    assert jlax.collapse(a, 0, 1).shape is not None
    assert jlax.conv_dimension_numbers(None, None, "NCHW") == "NCHW"

    assert jlax.dynamic_index_in_dim(a, 0).shape is not None
    assert jlax.dynamic_slice_in_dim(a, 0, 1).shape is not None
    assert jlax.dynamic_update_index_in_dim(a, 0, 0).shape is not None
    assert jlax.dynamic_update_slice_in_dim(a, jnp.array([0]), 0).shape is not None

    # Just running to make sure they are mocked properly without failure
    jlax.fori_loop(0, 1, lambda i, val: val, 0)
    jlax.igamma_grad_a(a, b)
    jlax.index_in_dim(a, 0)
    jlax.index_take(a, [0], [0])
    jlax.infeed()

    assert jlax.logistic(a).shape is not None
    jlax.outfeed(a)

    assert jlax.pbroadcast(a, "x").shape is not None
    assert jlax.pdot(a, a, "x").shape is not None
    assert jlax.pmax(a, "x").shape is not None
    assert jlax.pmin(a, "x").shape is not None
    assert jlax.population_count(a).shape is not None
    assert jlax.ppermute(a, "x", None).shape is not None
    assert jlax.pshuffle(a, "x", None).shape is not None
    assert jlax.psum_scatter(a, "x").shape is not None
    assert jlax.pswapaxes(a, "x", 0).shape is not None
    assert jlax.ragged_dot(a, a).shape is not None
    assert jlax.random_gamma_grad(a, b).shape is not None
    assert jlax.reduce_precision(a, 5, 10).shape is not None
    assert jlax.rev(a, [0]).shape is not None
    assert jlax.scan_bind(None) is None

    assert jlax.scatter_apply(a, None, None, None, None).shape is not None
    assert jlax.scatter_max(a, None, None, None).shape is not None
    assert jlax.scatter_min(a, None, None, None).shape is not None
    assert jlax.scatter_mul(a, None, None, None).shape is not None

    assert jlax.select_n(jnp.array([True, False, True]), a, b).shape is not None
    assert jlax.slice_in_dim(a, 0, 1).shape is not None
    assert jlax.sort_key_val(a, b)[0].shape is not None
    assert jlax.top_k(a, 1)[0].shape is not None

    jlax.while_loop(lambda x: False, lambda x: x, 0)
    assert jlax.with_sharding_constraint(a, None).shape is not None

    assert jlax.xeinsum("i,i->", a, a).shape is not None
    assert jlax.zeros_like_array(a).shape is not None

    assert jlax.after_all() is None
    assert jlax.all_gather(a, "x").shape is not None
    assert jlax.all_to_all(a, "x", 0, 0).shape is not None
    assert jlax.axis_index("x") == 0
    assert jlax.batch_matmul(a, a).shape is not None
    assert jlax.clz(a).shape is not None
    assert jlax.complex(a, a).shape is not None
    assert jlax.conv(a, a, None, None).shape is not None
    assert jlax.convert_element_type(a, np.float32).shape is not None
    assert jlax.create_token() is None
    assert jlax.cumlogsumexp(a).shape is not None
    assert jlax.cummax(a).shape is not None
    assert jlax.cummin(a).shape is not None
    assert jlax.cumprod(a).shape is not None
    assert jlax.integer_pow(a, 2).shape is not None
    assert jlax.is_finite(a).shape is not None
    assert jlax.map(lambda x: x, a).shape is not None
    assert jlax.pow(a, 2).shape is not None
    assert jlax.rem(a, 2).shape is not None
    assert jlax.switch(0, [lambda x: x], a).shape is not None


# Covered ops: Pbroadcast Pdot Ppermute Pshuffle
