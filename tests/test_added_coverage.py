import pytest


def test_partial_eval_config():
    """Test partial_eval config."""
    import zero_jax.interpreters.partial_eval as pe

    assert isinstance(pe.partial_eval_jaxpr_custom_rules, dict)

    from zero_jax.interpreters.partial_eval.config import Config

    c = Config()
    assert c is not None


def test_traceback_util_config_functions():
    """Test traceback_util config functions."""
    import zero_jax.interpreters.traceback_util.config as tuc

    funcs = [
        "already_configured_with_absl",
        "bcoo_cusparse_lowering",
        "cast",
        "compilation_cache_dir",
        "compilation_cache_include_metadata_in_key",
        "compilation_cache_max_size",
        "config",
        "custom_vjp_disable_shape_check",
        "debug_infs",
        "debug_nans",
        "default_matmul_precision",
        "define_optional_enum_state",
        "define_optional_string_state",
        "define_string_or_object_state",
        "enable_compilation_cache",
        "enable_custom_vjp_by_custom_transpose",
        "explicit_device_get_scope",
        "explicit_device_put_scope",
        "include_full_tracebacks_in_locations",
        "jax2tf_associative_scan_reductions",
        "jax2tf_default_native_serialization",
        "jax_export_calling_convention_version",
        "jax_serialization_version",
        "log_checkpoint_residuals",
        "numpy_dtype_promotion",
        "parse_flags_with_absl",
        "persistent_cache_min_compile_time_secs",
        "persistent_cache_min_entry_size_bytes",
        "pgle_aggregation_percentile",
        "pmap_no_rank_reduction",
        "raise_persistent_cache_errors",
        "share_autotune_config_between_hosts",
        "share_binary_between_hosts",
        "share_binary_between_hosts_timeout_ms",
        "threefry_gpu_kernel_lowering",
        "threefry_partitionable",
        "traceback_in_locations_limit",
        "transfer_guard_device_to_device",
        "transfer_guard_device_to_host",
        "transfer_guard_host_to_device",
        "update_thread_local_jit_state",
    ]
    for func in funcs:
        getattr(tuc, func)()
    assert tuc.annotations is None


def test_traceback_util_util_config_functions():
    """Test traceback_util.util config functions."""
    import zero_jax.interpreters.traceback_util.util.config as tuuc

    funcs = [
        "already_configured_with_absl",
        "annotations",
        "bcoo_cusparse_lowering",
        "cast",
        "check_tracer_leaks",
        "compilation_cache_dir",
        "compilation_cache_include_metadata_in_key",
        "compilation_cache_max_size",
        "config",
        "custom_vjp_disable_shape_check",
        "debug_infs",
        "debug_nans",
        "default_dtype_bits",
        "default_matmul_precision",
        "default_prng_impl",
        "define_bool_state",
        "define_enum_state",
        "define_float_state",
        "define_int_state",
        "define_optional_enum_state",
        "define_optional_string_state",
        "define_string_or_object_state",
        "define_string_state",
        "distributed_debug",
        "enable_compilation_cache",
        "enable_custom_prng",
        "enable_custom_vjp_by_custom_transpose",
        "explain_cache_misses",
        "explicit_device_get_scope",
        "explicit_device_put_scope",
        "include_full_tracebacks_in_locations",
        "jax2tf_associative_scan_reductions",
        "jax2tf_default_native_serialization",
        "jax_export_calling_convention_version",
        "jax_serialization_version",
        "log_checkpoint_residuals",
        "numpy_dtype_promotion",
        "numpy_rank_promotion",
        "parse_flags_with_absl",
        "persistent_cache_min_compile_time_secs",
        "persistent_cache_min_entry_size_bytes",
        "pgle_aggregation_percentile",
        "pgle_profiling_runs",
        "pmap_no_rank_reduction",
        "pmap_shmap_merge",
        "raise_persistent_cache_errors",
        "random_seed_offset",
        "remat_opt_barrier",
        "share_autotune_config_between_hosts",
        "share_binary_between_hosts",
        "share_binary_between_hosts_timeout_ms",
        "softmax_custom_jvp",
        "threefry_gpu_kernel_lowering",
        "threefry_partitionable",
        "traceback_filtering",
        "traceback_in_locations_limit",
        "transfer_guard_device_to_device",
        "transfer_guard_device_to_host",
        "transfer_guard_host_to_device",
        "update_thread_local_jit_state",
    ]
    for func in funcs:
        fn = getattr(tuuc, func)
        if callable(fn):
            fn()
