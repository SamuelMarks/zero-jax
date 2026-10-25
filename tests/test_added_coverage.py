"""Tests for tests/test_added_coverage.py."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, VariadicCallable


def test_partial_eval_config() -> None:
    """Test partial_eval config."""
    import zero_jax.interpreters.partial_eval as pe

    assert isinstance(pe.partial_eval_jaxpr_custom_rules, dict)

    from zero_jax.interpreters.partial_eval.config import Config

    c = Config()
    assert c is not None


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
    assert tuc.annotations is None


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
