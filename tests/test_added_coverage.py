"""Tests for tests/test_added_coverage.py."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, VariadicCallable


def test_partial_eval_config() -> None:
    """Test partial_eval config."""
    import zero_jax.interpreters.partial_eval as pe

    assert isinstance(pe.partial_eval_jaxpr_custom_rules, dict)

    from zero_jax.interpreters.partial_eval.config import Config

    c = Config()
    assert isinstance(c, Config)


def test_traceback_util_config_functions() -> None:
    """Test traceback_util config functions."""
    import zero_jax.interpreters.traceback_util.config as tuc

    funcs: list[VariadicCallable[ArrayLikeSpec]] = [
        tuc.already_configured_with_absl,
        tuc.bcoo_cusparse_lowering,
        tuc.cast,
        tuc.compilation_cache_dir,
        tuc.compilation_cache_include_metadata_in_key,
        tuc.compilation_cache_max_size,
        tuc.config,
        tuc.custom_vjp_disable_shape_check,
        tuc.debug_infs,
        tuc.debug_nans,
        tuc.default_matmul_precision,
        tuc.define_optional_enum_state,
        tuc.define_optional_string_state,
        tuc.define_string_or_object_state,
        tuc.enable_compilation_cache,
        tuc.enable_custom_vjp_by_custom_transpose,
        tuc.explicit_device_get_scope,
        tuc.explicit_device_put_scope,
        tuc.include_full_tracebacks_in_locations,
        tuc.jax2tf_associative_scan_reductions,
        tuc.jax2tf_default_native_serialization,
        tuc.jax_export_calling_convention_version,
        tuc.jax_serialization_version,
        tuc.log_checkpoint_residuals,
        tuc.numpy_dtype_promotion,
        tuc.parse_flags_with_absl,
        tuc.persistent_cache_min_compile_time_secs,
        tuc.persistent_cache_min_entry_size_bytes,
        tuc.pgle_aggregation_percentile,
        tuc.pmap_no_rank_reduction,
        tuc.raise_persistent_cache_errors,
        tuc.share_autotune_config_between_hosts,
        tuc.share_binary_between_hosts,
        tuc.share_binary_between_hosts_timeout_ms,
        tuc.threefry_gpu_kernel_lowering,
        tuc.threefry_partitionable,
        tuc.traceback_in_locations_limit,
        tuc.transfer_guard_device_to_device,
        tuc.transfer_guard_device_to_host,
        tuc.transfer_guard_host_to_device,
        tuc.update_thread_local_jit_state,
    ]
    for fn in funcs:
        fn()
    assert tuc.annotations == "__future__.annotations"


def test_traceback_util_util_config_functions() -> None:
    """Test traceback_util.util config functions."""
    import zero_jax.interpreters.traceback_util.util.config as tuuc

    funcs: list[ArrayLikeSpec] = [
        tuuc.already_configured_with_absl,
        tuuc.annotations,
        tuuc.bcoo_cusparse_lowering,
        tuuc.cast,
        tuuc.check_tracer_leaks,
        tuuc.compilation_cache_dir,
        tuuc.compilation_cache_include_metadata_in_key,
        tuuc.compilation_cache_max_size,
        tuuc.config,
        tuuc.custom_vjp_disable_shape_check,
        tuuc.debug_infs,
        tuuc.debug_nans,
        tuuc.default_dtype_bits,
        tuuc.default_matmul_precision,
        tuuc.default_prng_impl,
        tuuc.define_bool_state,
        tuuc.define_enum_state,
        tuuc.define_float_state,
        tuuc.define_int_state,
        tuuc.define_optional_enum_state,
        tuuc.define_optional_string_state,
        tuuc.define_string_or_object_state,
        tuuc.define_string_state,
        tuuc.distributed_debug,
        tuuc.enable_compilation_cache,
        tuuc.enable_custom_prng,
        tuuc.enable_custom_vjp_by_custom_transpose,
        tuuc.explain_cache_misses,
        tuuc.explicit_device_get_scope,
        tuuc.explicit_device_put_scope,
        tuuc.include_full_tracebacks_in_locations,
        tuuc.jax2tf_associative_scan_reductions,
        tuuc.jax2tf_default_native_serialization,
        tuuc.jax_export_calling_convention_version,
        tuuc.jax_serialization_version,
        tuuc.log_checkpoint_residuals,
        tuuc.numpy_dtype_promotion,
        tuuc.numpy_rank_promotion,
        tuuc.parse_flags_with_absl,
        tuuc.persistent_cache_min_compile_time_secs,
        tuuc.persistent_cache_min_entry_size_bytes,
        tuuc.pgle_aggregation_percentile,
        tuuc.pgle_profiling_runs,
        tuuc.pmap_no_rank_reduction,
        tuuc.pmap_shmap_merge,
        tuuc.raise_persistent_cache_errors,
        tuuc.random_seed_offset,
        tuuc.remat_opt_barrier,
        tuuc.share_autotune_config_between_hosts,
        tuuc.share_binary_between_hosts,
        tuuc.share_binary_between_hosts_timeout_ms,
        tuuc.softmax_custom_jvp,
        tuuc.threefry_gpu_kernel_lowering,
        tuuc.threefry_partitionable,
        tuuc.traceback_filtering,
        tuuc.traceback_in_locations_limit,
        tuuc.transfer_guard_device_to_device,
        tuuc.transfer_guard_device_to_host,
        tuuc.transfer_guard_host_to_device,
        tuuc.update_thread_local_jit_state,
    ]
    for fn in funcs:
        if callable(fn):
            fn()


def test_decomposed_submodule_helpers() -> None:
    """Test helpers and branches across decomposed lax and numpy submodules."""
    import pytest

    import zero_jax.numpy as jnp
    from zero_jax.lax._bitwise_math import regularized_incomplete_beta
    from zero_jax.lax._collectives import pbkdf2
    from zero_jax.numpy._ndarray import _ensure_tensor

    beta_res = regularized_incomplete_beta(jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.5]))
    assert beta_res.shape == (1,)

    key_res = pbkdf2(jnp.array([1, 2], dtype=jnp.uint8), jnp.array([3, 4], dtype=jnp.uint8), iterations=10, key_length=8)
    assert key_res.shape == (8,)

    t = _ensure_tensor(jnp.array([1.0, 2.0]))
    assert t.shape == (2,)

    rstack = jnp.row_stack([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
    assert rstack.shape == (2, 2)

    in1d_res = jnp.in1d(jnp.array([1, 2, 3]), jnp.array([2, 4]))
    assert in1d_res.shape == (3,)

    tiled = jnp.tile(jnp.array([1.0, 2.0]), jnp.array([2]))
    assert tiled.shape == (4,)

    import zero_jax
    import zero_jax.dlpack as jdl
    import zero_jax.dtypes as jdtypes

    dl_arr = jnp.from_dlpack(jdl.to_dlpack(jnp.array([1.0, 2.0])))
    assert dl_arr.shape == (2,)

    assert jdtypes.finfo(jdtypes.float8_e4m3fn).bits == 8
    assert jdtypes.iinfo(jnp.uint4).bits == 4
    assert jdtypes.iinfo(jnp.int4).bits == 4
    assert jdtypes.iinfo(jnp.uint16).max == 65535
    assert jdtypes.iinfo(jnp.uint64).max == 18446744073709551615

    # put / place / put_along_axis modes and inplace validation
    arr_p = jnp.array([1.0, 2.0, 3.0])
    assert jnp.put(arr_p, 5, 9.0, mode="wrap")[2] == 9.0
    assert jnp.put(arr_p, 5, 8.0, mode="clip")[2] == 8.0
    with pytest.raises(TypeError, match="inplace=False"):
        jnp.put(arr_p, 0, 1.0, inplace=True)
    with pytest.raises(TypeError, match="inplace=False"):
        jnp.put_along_axis(arr_p, jnp.array([0]), jnp.array([1.0]), axis=0, inplace=True)
    with pytest.raises(TypeError, match="inplace=False"):
        jnp.place(arr_p, jnp.array([True, False, False]), jnp.array([1.0]), inplace=True)

    # Traced _IndexUpdateRef 2D and operations (add, multiply, divide, power, min, max)
    @zero_jax.jit
    def _traced_updates(x: jnp.ndarray) -> jnp.ndarray:
        """Apply traced index updates on a 2D array.

        Args:
            x: Input 2D array.

        Returns:
            jnp.ndarray: Updated 2D array.
        """
        y = x.at[0].set(2.0)
        y = y.at[0].add(1.0)
        y = y.at[0].multiply(2.0)
        y = y.at[0].divide(2.0)
        y = y.at[0].power(2.0)
        y = y.at[0].min(5.0)
        return y.at[0].max(10.0)

    res_2d = _traced_updates(jnp.ones((2, 2)))
    assert res_2d.shape == (2, 2)

    # Multi-output grad, vmap in_axes / out_axes, jit static_argnums, jvp empty primals
    g_multi = zero_jax.grad(lambda x: (x * 2.0, x * 3.0))(jnp.array(1.0))
    assert g_multi.item() == 5.0

    v_in = zero_jax.vmap(lambda x, y: x + y, in_axes=1, out_axes=1)(jnp.ones((2, 3)), jnp.ones((2, 3)))
    assert v_in.shape == (2, 3)
    v_tuple = zero_jax.vmap(lambda x, y: x + y, in_axes=(1, 0))(jnp.ones((2, 3)), jnp.ones((3, 2)))
    assert v_tuple.shape == (3, 2)

    jit_static = zero_jax.jit(lambda x, scale: x * scale, static_argnums=1)
    assert jit_static(jnp.array(3.0), 4.0).item() == 12.0
    assert zero_jax.jit(lambda: 42)() == 42

    p_empty, t_empty = zero_jax.jvp(lambda: jnp.array(3.0), (), ())
    assert p_empty.item() == 3.0
    assert t_empty.item() == 0.0
