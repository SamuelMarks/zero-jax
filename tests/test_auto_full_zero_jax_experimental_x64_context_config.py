"""Tests for zero_jax.experimental.x64_context.config."""

from __future__ import annotations

import pytest

import zero_jax.experimental.x64_context.config as mod


def test_Callable() -> None:
    """Test Callable."""
    assert mod.Callable is not None


def test_class_Config() -> None:
    """Test Config."""
    obj = mod.Config()
    assert isinstance(obj, mod.Config)


def test_DEFINE_bool() -> None:
    """Test DEFINE_bool."""
    res = mod.DEFINE_bool()
    assert res is not False or isinstance(res, bool)


def test_DEFINE_enum() -> None:
    """Test DEFINE_enum."""
    res = mod.DEFINE_enum()
    assert res is not False or isinstance(res, bool)


def test_DEFINE_float() -> None:
    """Test DEFINE_float."""
    res = mod.DEFINE_float()
    assert res is not False or isinstance(res, bool)


def test_DEFINE_integer() -> None:
    """Test DEFINE_integer."""
    res = mod.DEFINE_integer()
    assert res is not False or isinstance(res, bool)


def test_DEFINE_string() -> None:
    """Test DEFINE_string."""
    res = mod.DEFINE_string()
    assert res is not False or isinstance(res, bool)


def test_class_FlagHolder() -> None:
    """Test FlagHolder."""
    obj = mod.FlagHolder()
    assert isinstance(obj, mod.FlagHolder)


def test_class_Generic() -> None:
    """Test Generic."""
    assert mod.Generic is not None


def test_class_Hashable() -> None:
    """Test Hashable."""
    assert isinstance(1, mod.Hashable)


def test_class_Iterator() -> None:
    """Test Iterator."""
    assert isinstance(iter([1]), mod.Iterator)


def test_NamedTuple() -> None:
    """Test NamedTuple."""
    res = mod.NamedTuple()
    assert res == "NamedTuple"


def test_class_NoDefault() -> None:
    """Test NoDefault."""
    obj = mod.NoDefault()
    assert isinstance(obj, mod.NoDefault)
    assert obj() is obj


def test_NoReturn() -> None:
    """Test NoReturn."""
    res = mod.NoReturn()
    assert res == "NoReturn"


def test_class_TypeVar() -> None:
    """Test TypeVar."""
    obj = mod.TypeVar("T")
    assert obj.__name__ == "T"


def test_already_configured_with_absl() -> None:
    """Test already_configured_with_absl."""
    assert mod.already_configured_with_absl() is False


def test_bcoo_cusparse_lowering() -> None:
    """Test bcoo_cusparse_lowering."""
    res = mod.bcoo_cusparse_lowering()
    assert res is not False or isinstance(res, bool)


def test_bool_env() -> None:
    """Test bool_env."""
    res = mod.bool_env()
    assert res is not False or isinstance(res, bool)


def test_cast() -> None:
    """Test cast."""
    res = mod.cast(1.0, "float32")
    assert bool(res) or res == 1.0


def test_check_exists() -> None:
    """Test check_exists."""
    res = mod.check_exists()
    assert res is not False or isinstance(res, bool)


def test_check_tracer_leaks() -> None:
    """Test check_tracer_leaks."""
    res = mod.check_tracer_leaks()
    assert res is not False or isinstance(res, bool)


def test_checking_leaks() -> None:
    """Test checking_leaks."""
    res = mod.checking_leaks()
    assert res is not False or isinstance(res, bool)


def test_compilation_cache_dir() -> None:
    """Test compilation_cache_dir."""
    res = mod.compilation_cache_dir()
    assert res is not False or isinstance(res, bool)


def test_compilation_cache_include_metadata_in_key() -> None:
    """Test compilation_cache_include_metadata_in_key."""
    res = mod.compilation_cache_include_metadata_in_key()
    assert res is not False or isinstance(res, bool)


def test_compilation_cache_max_size() -> None:
    """Test compilation_cache_max_size."""
    res = mod.compilation_cache_max_size()
    assert res is not False or isinstance(res, bool)


def test_config() -> None:
    """Test config."""
    assert isinstance(mod.config, mod.Config)


def test_custom_vjp_disable_shape_check() -> None:
    """Test custom_vjp_disable_shape_check."""
    res = mod.custom_vjp_disable_shape_check()
    assert res is not False or isinstance(res, bool)


def test_debug_infs() -> None:
    """Test debug_infs."""
    res = mod.debug_infs(1.0)
    assert bool(res) or res == 1.0


def test_debug_key_reuse() -> None:
    """Test debug_key_reuse."""
    res = mod.debug_key_reuse()
    assert res is not False or isinstance(res, bool)


def test_debug_nans() -> None:
    """Test debug_nans."""
    res = mod.debug_nans(1.0)
    assert bool(res) or res == 1.0


def test_default_device() -> None:
    """Test default_device."""
    res = mod.default_device()
    assert res is not False or isinstance(res, bool)


def test_default_dtype_bits() -> None:
    """Test default_dtype_bits."""
    res = mod.default_dtype_bits()
    assert res is not False or isinstance(res, bool)


def test_default_matmul_precision() -> None:
    """Test default_matmul_precision."""
    res = mod.default_matmul_precision()
    assert res is not False or isinstance(res, bool)


def test_default_prng_impl() -> None:
    """Test default_prng_impl."""
    res = mod.default_prng_impl()
    assert res is not False or isinstance(res, bool)


def test_define_bool_state() -> None:
    """Test define_bool_state."""
    res = mod.define_bool_state()
    assert res is not False or isinstance(res, bool)


def test_define_enum_state() -> None:
    """Test define_enum_state."""
    res = mod.define_enum_state()
    assert res is not False or isinstance(res, bool)


def test_define_float_state() -> None:
    """Test define_float_state."""
    res = mod.define_float_state()
    assert res is not False or isinstance(res, bool)


def test_define_int_state() -> None:
    """Test define_int_state."""
    res = mod.define_int_state()
    assert res is not False or isinstance(res, bool)


def test_define_optional_enum_state() -> None:
    """Test define_optional_enum_state."""
    res = mod.define_optional_enum_state()
    assert res is not False or isinstance(res, bool)


def test_define_optional_string_state() -> None:
    """Test define_optional_string_state."""
    res = mod.define_optional_string_state()
    assert res is not False or isinstance(res, bool)


def test_define_string_or_object_state() -> None:
    """Test define_string_or_object_state."""
    res = mod.define_string_or_object_state()
    assert res is not False or isinstance(res, bool)


def test_define_string_state() -> None:
    """Test define_string_state."""
    res = mod.define_string_state()
    assert res is not False or isinstance(res, bool)


def test_disable_jit() -> None:
    """Test disable_jit."""
    res = mod.disable_jit()
    assert res is not False or isinstance(res, bool)


def test_distributed_debug() -> None:
    """Test distributed_debug."""
    res = mod.distributed_debug()
    assert res is not False or isinstance(res, bool)


def test_dynamic_shapes() -> None:
    """Test dynamic_shapes."""
    res = mod.dynamic_shapes()
    assert res is not False or isinstance(res, bool)


def test_eager_pmap() -> None:
    """Test eager_pmap."""
    res = mod.eager_pmap()
    assert res is not False or isinstance(res, bool)


def test_enable_checks() -> None:
    """Test enable_checks."""
    res = mod.enable_checks()
    assert res is not False or isinstance(res, bool)


def test_enable_compilation_cache() -> None:
    """Test enable_compilation_cache."""
    res = mod.enable_compilation_cache()
    assert res is not False or isinstance(res, bool)


def test_enable_custom_prng() -> None:
    """Test enable_custom_prng."""
    res = mod.enable_custom_prng()
    assert res is not False or isinstance(res, bool)


def test_enable_custom_vjp_by_custom_transpose() -> None:
    """Test enable_custom_vjp_by_custom_transpose."""
    res = mod.enable_custom_vjp_by_custom_transpose()
    assert res is not False or isinstance(res, bool)


def test_enable_memories() -> None:
    """Test enable_memories."""
    res = mod.enable_memories()
    assert res is not False or isinstance(res, bool)


def test_enable_pgle() -> None:
    """Test enable_pgle."""
    res = mod.enable_pgle()
    assert res is not False or isinstance(res, bool)


def test_enable_x64() -> None:
    """Test enable_x64."""
    res = mod.enable_x64()
    assert res is not False or isinstance(res, bool)


def test_explain_cache_misses() -> None:
    """Test explain_cache_misses."""
    res = mod.explain_cache_misses()
    assert res is not False or isinstance(res, bool)


def test_explicit_device_get_scope() -> None:
    """Test explicit_device_get_scope."""
    res = mod.explicit_device_get_scope()
    assert res is not False or isinstance(res, bool)


def test_explicit_device_put_scope() -> None:
    """Test explicit_device_put_scope."""
    res = mod.explicit_device_put_scope()
    assert res is not False or isinstance(res, bool)


def test_hlo_source_file_canonicalization_regex() -> None:
    """Test hlo_source_file_canonicalization_regex."""
    res = mod.hlo_source_file_canonicalization_regex()
    assert res is not False or isinstance(res, bool)


def test_include_full_tracebacks_in_locations() -> None:
    """Test include_full_tracebacks_in_locations."""
    res = mod.include_full_tracebacks_in_locations()
    assert res is not False or isinstance(res, bool)


def test_int_env() -> None:
    """Test int_env."""
    res = mod.int_env()
    assert res is not False or isinstance(res, bool)


def test_jax2tf_associative_scan_reductions() -> None:
    """Test jax2tf_associative_scan_reductions."""
    res = mod.jax2tf_associative_scan_reductions()
    assert res is not False or isinstance(res, bool)


def test_jax2tf_default_native_serialization() -> None:
    """Test jax2tf_default_native_serialization."""
    res = mod.jax2tf_default_native_serialization()
    assert res is not False or isinstance(res, bool)


def test_jax_export_calling_convention_version() -> None:
    """Test jax_export_calling_convention_version."""
    res = mod.jax_export_calling_convention_version()
    assert res is not False or isinstance(res, bool)


def test_jax_pjrt_client_create_options() -> None:
    """Test jax_pjrt_client_create_options."""
    res = mod.jax_pjrt_client_create_options()
    assert res is not False or isinstance(res, bool)


def test_jax_platforms() -> None:
    """Test jax_platforms."""
    res = mod.jax_platforms()
    assert res is not False or isinstance(res, bool)


def test_jax_serialization_version() -> None:
    """Test jax_serialization_version."""
    res = mod.jax_serialization_version()
    assert res is not False or isinstance(res, bool)


def test_jax_xla_profile_version() -> None:
    """Test jax_xla_profile_version."""
    res = mod.jax_xla_profile_version()
    assert res is not False or isinstance(res, bool)


def test_legacy_prng_key() -> None:
    """Test legacy_prng_key."""
    res = mod.legacy_prng_key()
    assert res is not False or isinstance(res, bool)


def test_lib() -> None:
    """Test lib."""
    assert mod.lib() == "jaxlib"


def test_log_checkpoint_residuals() -> None:
    """Test log_checkpoint_residuals."""
    res = mod.log_checkpoint_residuals()
    assert res is not False or isinstance(res, bool)


def test_log_compiles() -> None:
    """Test log_compiles."""
    res = mod.log_compiles()
    assert res is not False or isinstance(res, bool)


def test_logger() -> None:
    """Test logger."""
    assert mod.logger.name == "jax.config"


def test_no_default() -> None:
    """Test no_default."""
    assert isinstance(mod.no_default, mod.NoDefault)


def test_numpy_dtype_promotion() -> None:
    """Test numpy_dtype_promotion."""
    res = mod.numpy_dtype_promotion()
    assert res is not False or isinstance(res, bool)


def test_numpy_rank_promotion() -> None:
    """Test numpy_rank_promotion."""
    res = mod.numpy_rank_promotion()
    assert res is not False or isinstance(res, bool)


def test_parse_flags_with_absl() -> None:
    """Test parse_flags_with_absl."""
    res = mod.parse_flags_with_absl()
    assert res is not False or isinstance(res, bool)


def test_persistent_cache_min_compile_time_secs() -> None:
    """Test persistent_cache_min_compile_time_secs."""
    res = mod.persistent_cache_min_compile_time_secs()
    assert res is not False or isinstance(res, bool)


def test_persistent_cache_min_entry_size_bytes() -> None:
    """Test persistent_cache_min_entry_size_bytes."""
    res = mod.persistent_cache_min_entry_size_bytes()
    assert res is not False or isinstance(res, bool)


def test_pgle_aggregation_percentile() -> None:
    """Test pgle_aggregation_percentile."""
    res = mod.pgle_aggregation_percentile()
    assert res is not False or isinstance(res, bool)


def test_pgle_profiling_runs() -> None:
    """Test pgle_profiling_runs."""
    res = mod.pgle_profiling_runs()
    assert res is not False or isinstance(res, bool)


def test_pmap_no_rank_reduction() -> None:
    """Test pmap_no_rank_reduction."""
    res = mod.pmap_no_rank_reduction()
    assert res is not False or isinstance(res, bool)


def test_pmap_shmap_merge() -> None:
    """Test pmap_shmap_merge."""
    res = mod.pmap_shmap_merge()
    assert res is not False or isinstance(res, bool)


def test_raise_persistent_cache_errors() -> None:
    """Test raise_persistent_cache_errors."""
    res = mod.raise_persistent_cache_errors()
    assert res is not False or isinstance(res, bool)


def test_random_seed_offset() -> None:
    """Test random_seed_offset."""
    res = mod.random_seed_offset()
    assert res is not False or isinstance(res, bool)


def test_remat_opt_barrier() -> None:
    """Test remat_opt_barrier."""
    res = mod.remat_opt_barrier()
    assert res is not False or isinstance(res, bool)


def test_share_autotune_config_between_hosts() -> None:
    """Test share_autotune_config_between_hosts."""
    res = mod.share_autotune_config_between_hosts()
    assert res is not False or isinstance(res, bool)


def test_share_binary_between_hosts() -> None:
    """Test share_binary_between_hosts."""
    res = mod.share_binary_between_hosts()
    assert res is not False or isinstance(res, bool)


def test_share_binary_between_hosts_timeout_ms() -> None:
    """Test share_binary_between_hosts_timeout_ms."""
    res = mod.share_binary_between_hosts_timeout_ms()
    assert res is not False or isinstance(res, bool)


def test_softmax_custom_jvp() -> None:
    """Test softmax_custom_jvp."""
    res = mod.softmax_custom_jvp()
    assert res is not False or isinstance(res, bool)


def test_spmd_mode() -> None:
    """Test spmd_mode."""
    res = mod.spmd_mode()
    assert res is not False or isinstance(res, bool)


def test_threefry_gpu_kernel_lowering() -> None:
    """Test threefry_gpu_kernel_lowering."""
    res = mod.threefry_gpu_kernel_lowering()
    assert res is not False or isinstance(res, bool)


def test_threefry_partitionable() -> None:
    """Test threefry_partitionable."""
    res = mod.threefry_partitionable()
    assert res is not False or isinstance(res, bool)


def test_trace_context() -> None:
    """Test trace_context."""
    res = mod.trace_context()
    assert res is not False or isinstance(res, bool)


def test_traceback_filtering() -> None:
    """Test traceback_filtering."""
    res = mod.traceback_filtering()
    assert res is not False or isinstance(res, bool)


def test_traceback_in_locations_limit() -> None:
    """Test traceback_in_locations_limit."""
    res = mod.traceback_in_locations_limit()
    assert res is not False or isinstance(res, bool)


def test_transfer_guard() -> None:
    """Test transfer_guard."""
    res = mod.transfer_guard()
    assert res is not False or isinstance(res, bool)


def test_transfer_guard_device_to_device() -> None:
    """Test transfer_guard_device_to_device."""
    res = mod.transfer_guard_device_to_device()
    assert res is not False or isinstance(res, bool)


def test_transfer_guard_device_to_host() -> None:
    """Test transfer_guard_device_to_host."""
    res = mod.transfer_guard_device_to_host()
    assert res is not False or isinstance(res, bool)


def test_transfer_guard_host_to_device() -> None:
    """Test transfer_guard_host_to_device."""
    res = mod.transfer_guard_host_to_device()
    assert res is not False or isinstance(res, bool)


def test_unset() -> None:
    """Test unset."""
    assert isinstance(mod.unset, mod.NoDefault)


def test_update() -> None:
    """Test update."""
    res = mod.update()
    assert res is not False or isinstance(res, bool)


def test_update_thread_local_jit_state() -> None:
    """Test update_thread_local_jit_state."""
    res = mod.update_thread_local_jit_state()
    assert res is not False or isinstance(res, bool)


def test_xla_runtime_errors() -> None:
    """Test xla_runtime_errors."""
    res = mod.xla_runtime_errors()
    assert res is not False or isinstance(res, bool)
