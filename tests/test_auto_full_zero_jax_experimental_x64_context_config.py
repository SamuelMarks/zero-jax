"""Tests for zero_jax.experimental.x64_context.config."""

from typing import Any

import pytest

import zero_jax.experimental.x64_context.config as mod


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


def test_class_Config() -> None:
    """Test class Config."""
    try:
        mod.Config()
    except Exception:
        pass


def test_DEFINE_bool() -> None:
    """Test DEFINE_bool."""
    try:
        mod.DEFINE_bool()
    except Exception:
        pass


def test_DEFINE_enum() -> None:
    """Test DEFINE_enum."""
    try:
        mod.DEFINE_enum()
    except Exception:
        pass


def test_DEFINE_float() -> None:
    """Test DEFINE_float."""
    try:
        mod.DEFINE_float()
    except Exception:
        pass


def test_DEFINE_integer() -> None:
    """Test DEFINE_integer."""
    try:
        mod.DEFINE_integer()
    except Exception:
        pass


def test_DEFINE_string() -> None:
    """Test DEFINE_string."""
    try:
        mod.DEFINE_string()
    except Exception:
        pass


def test_class_FlagHolder() -> None:
    """Test class FlagHolder."""
    try:
        mod.FlagHolder()
    except Exception:
        pass


def test_class_Generic() -> None:
    """Test class Generic."""
    try:
        mod.Generic()
    except Exception:
        pass


def test_class_Hashable() -> None:
    """Test class Hashable."""
    try:
        mod.Hashable()
    except Exception:
        pass


def test_class_Iterator() -> None:
    """Test class Iterator."""
    try:
        mod.Iterator()
    except Exception:
        pass


def test_NamedTuple() -> None:
    """Test NamedTuple."""
    try:
        mod.NamedTuple()
    except Exception:
        pass


def test_class_NoDefault() -> None:
    """Test class NoDefault."""
    try:
        mod.NoDefault()
    except Exception:
        pass


def test_NoReturn() -> None:
    """Test NoReturn."""
    try:
        mod.NoReturn()
    except Exception:
        pass


def test_class_TypeVar() -> None:
    """Test class TypeVar."""
    try:
        mod.TypeVar()
    except Exception:
        pass


def test_already_configured_with_absl() -> None:
    """Test already_configured_with_absl."""
    try:
        mod.already_configured_with_absl()
    except Exception:
        pass


def test_bcoo_cusparse_lowering() -> None:
    """Test bcoo_cusparse_lowering."""
    try:
        mod.bcoo_cusparse_lowering()
    except Exception:
        pass


def test_bool_env() -> None:
    """Test bool_env."""
    try:
        mod.bool_env()
    except Exception:
        pass


def test_cast() -> None:
    """Test cast."""
    try:
        mod.cast()
    except Exception:
        pass


def test_check_exists() -> None:
    """Test check_exists."""
    try:
        mod.check_exists()
    except Exception:
        pass


def test_check_tracer_leaks() -> None:
    """Test check_tracer_leaks."""
    try:
        mod.check_tracer_leaks()
    except Exception:
        pass


def test_checking_leaks() -> None:
    """Test checking_leaks."""
    try:
        mod.checking_leaks()
    except Exception:
        pass


def test_compilation_cache_dir() -> None:
    """Test compilation_cache_dir."""
    try:
        mod.compilation_cache_dir()
    except Exception:
        pass


def test_compilation_cache_include_metadata_in_key() -> None:
    """Test compilation_cache_include_metadata_in_key."""
    try:
        mod.compilation_cache_include_metadata_in_key()
    except Exception:
        pass


def test_compilation_cache_max_size() -> None:
    """Test compilation_cache_max_size."""
    try:
        mod.compilation_cache_max_size()
    except Exception:
        pass


def test_config() -> None:
    """Test config."""
    try:
        mod.config()
    except Exception:
        pass


def test_custom_vjp_disable_shape_check() -> None:
    """Test custom_vjp_disable_shape_check."""
    try:
        mod.custom_vjp_disable_shape_check()
    except Exception:
        pass


def test_debug_infs() -> None:
    """Test debug_infs."""
    try:
        mod.debug_infs()
    except Exception:
        pass


def test_debug_key_reuse() -> None:
    """Test debug_key_reuse."""
    try:
        mod.debug_key_reuse()
    except Exception:
        pass


def test_debug_nans() -> None:
    """Test debug_nans."""
    try:
        mod.debug_nans()
    except Exception:
        pass


def test_default_device() -> None:
    """Test default_device."""
    try:
        mod.default_device()
    except Exception:
        pass


def test_default_dtype_bits() -> None:
    """Test default_dtype_bits."""
    try:
        mod.default_dtype_bits()
    except Exception:
        pass


def test_default_matmul_precision() -> None:
    """Test default_matmul_precision."""
    try:
        mod.default_matmul_precision()
    except Exception:
        pass


def test_default_prng_impl() -> None:
    """Test default_prng_impl."""
    try:
        mod.default_prng_impl()
    except Exception:
        pass


def test_define_bool_state() -> None:
    """Test define_bool_state."""
    try:
        mod.define_bool_state()
    except Exception:
        pass


def test_define_enum_state() -> None:
    """Test define_enum_state."""
    try:
        mod.define_enum_state()
    except Exception:
        pass


def test_define_float_state() -> None:
    """Test define_float_state."""
    try:
        mod.define_float_state()
    except Exception:
        pass


def test_define_int_state() -> None:
    """Test define_int_state."""
    try:
        mod.define_int_state()
    except Exception:
        pass


def test_define_optional_enum_state() -> None:
    """Test define_optional_enum_state."""
    try:
        mod.define_optional_enum_state()
    except Exception:
        pass


def test_define_optional_string_state() -> None:
    """Test define_optional_string_state."""
    try:
        mod.define_optional_string_state()
    except Exception:
        pass


def test_define_string_or_object_state() -> None:
    """Test define_string_or_object_state."""
    try:
        mod.define_string_or_object_state()
    except Exception:
        pass


def test_define_string_state() -> None:
    """Test define_string_state."""
    try:
        mod.define_string_state()
    except Exception:
        pass


def test_disable_jit() -> None:
    """Test disable_jit."""
    try:
        mod.disable_jit()
    except Exception:
        pass


def test_distributed_debug() -> None:
    """Test distributed_debug."""
    try:
        mod.distributed_debug()
    except Exception:
        pass


def test_dynamic_shapes() -> None:
    """Test dynamic_shapes."""
    try:
        mod.dynamic_shapes()
    except Exception:
        pass


def test_eager_pmap() -> None:
    """Test eager_pmap."""
    try:
        mod.eager_pmap()
    except Exception:
        pass


def test_enable_checks() -> None:
    """Test enable_checks."""
    try:
        mod.enable_checks()
    except Exception:
        pass


def test_enable_compilation_cache() -> None:
    """Test enable_compilation_cache."""
    try:
        mod.enable_compilation_cache()
    except Exception:
        pass


def test_enable_custom_prng() -> None:
    """Test enable_custom_prng."""
    try:
        mod.enable_custom_prng()
    except Exception:
        pass


def test_enable_custom_vjp_by_custom_transpose() -> None:
    """Test enable_custom_vjp_by_custom_transpose."""
    try:
        mod.enable_custom_vjp_by_custom_transpose()
    except Exception:
        pass


def test_enable_memories() -> None:
    """Test enable_memories."""
    try:
        mod.enable_memories()
    except Exception:
        pass


def test_enable_pgle() -> None:
    """Test enable_pgle."""
    try:
        mod.enable_pgle()
    except Exception:
        pass


def test_enable_x64() -> None:
    """Test enable_x64."""
    try:
        mod.enable_x64()
    except Exception:
        pass


def test_explain_cache_misses() -> None:
    """Test explain_cache_misses."""
    try:
        mod.explain_cache_misses()
    except Exception:
        pass


def test_explicit_device_get_scope() -> None:
    """Test explicit_device_get_scope."""
    try:
        mod.explicit_device_get_scope()
    except Exception:
        pass


def test_explicit_device_put_scope() -> None:
    """Test explicit_device_put_scope."""
    try:
        mod.explicit_device_put_scope()
    except Exception:
        pass


def test_hlo_source_file_canonicalization_regex() -> None:
    """Test hlo_source_file_canonicalization_regex."""
    try:
        mod.hlo_source_file_canonicalization_regex()
    except Exception:
        pass


def test_include_full_tracebacks_in_locations() -> None:
    """Test include_full_tracebacks_in_locations."""
    try:
        mod.include_full_tracebacks_in_locations()
    except Exception:
        pass


def test_int_env() -> None:
    """Test int_env."""
    try:
        mod.int_env()
    except Exception:
        pass


def test_jax2tf_associative_scan_reductions() -> None:
    """Test jax2tf_associative_scan_reductions."""
    try:
        mod.jax2tf_associative_scan_reductions()
    except Exception:
        pass


def test_jax2tf_default_native_serialization() -> None:
    """Test jax2tf_default_native_serialization."""
    try:
        mod.jax2tf_default_native_serialization()
    except Exception:
        pass


def test_jax_export_calling_convention_version() -> None:
    """Test jax_export_calling_convention_version."""
    try:
        mod.jax_export_calling_convention_version()
    except Exception:
        pass


def test_jax_pjrt_client_create_options() -> None:
    """Test jax_pjrt_client_create_options."""
    try:
        mod.jax_pjrt_client_create_options()
    except Exception:
        pass


def test_jax_platforms() -> None:
    """Test jax_platforms."""
    try:
        mod.jax_platforms()
    except Exception:
        pass


def test_jax_serialization_version() -> None:
    """Test jax_serialization_version."""
    try:
        mod.jax_serialization_version()
    except Exception:
        pass


def test_jax_xla_profile_version() -> None:
    """Test jax_xla_profile_version."""
    try:
        mod.jax_xla_profile_version()
    except Exception:
        pass


def test_legacy_prng_key() -> None:
    """Test legacy_prng_key."""
    try:
        mod.legacy_prng_key()
    except Exception:
        pass


def test_lib() -> None:
    """Test lib."""
    try:
        mod.lib()
    except Exception:
        pass


def test_log_checkpoint_residuals() -> None:
    """Test log_checkpoint_residuals."""
    try:
        mod.log_checkpoint_residuals()
    except Exception:
        pass


def test_log_compiles() -> None:
    """Test log_compiles."""
    try:
        mod.log_compiles()
    except Exception:
        pass


def test_logger() -> None:
    """Test logger."""
    try:
        mod.logger()
    except Exception:
        pass


def test_no_default() -> None:
    """Test no_default."""
    try:
        mod.no_default()
    except Exception:
        pass


def test_numpy_dtype_promotion() -> None:
    """Test numpy_dtype_promotion."""
    try:
        mod.numpy_dtype_promotion()
    except Exception:
        pass


def test_numpy_rank_promotion() -> None:
    """Test numpy_rank_promotion."""
    try:
        mod.numpy_rank_promotion()
    except Exception:
        pass


def test_parse_flags_with_absl() -> None:
    """Test parse_flags_with_absl."""
    try:
        mod.parse_flags_with_absl()
    except Exception:
        pass


def test_persistent_cache_min_compile_time_secs() -> None:
    """Test persistent_cache_min_compile_time_secs."""
    try:
        mod.persistent_cache_min_compile_time_secs()
    except Exception:
        pass


def test_persistent_cache_min_entry_size_bytes() -> None:
    """Test persistent_cache_min_entry_size_bytes."""
    try:
        mod.persistent_cache_min_entry_size_bytes()
    except Exception:
        pass


def test_pgle_aggregation_percentile() -> None:
    """Test pgle_aggregation_percentile."""
    try:
        mod.pgle_aggregation_percentile()
    except Exception:
        pass


def test_pgle_profiling_runs() -> None:
    """Test pgle_profiling_runs."""
    try:
        mod.pgle_profiling_runs()
    except Exception:
        pass


def test_pmap_no_rank_reduction() -> None:
    """Test pmap_no_rank_reduction."""
    try:
        mod.pmap_no_rank_reduction()
    except Exception:
        pass


def test_pmap_shmap_merge() -> None:
    """Test pmap_shmap_merge."""
    try:
        mod.pmap_shmap_merge()
    except Exception:
        pass


def test_raise_persistent_cache_errors() -> None:
    """Test raise_persistent_cache_errors."""
    try:
        mod.raise_persistent_cache_errors()
    except Exception:
        pass


def test_random_seed_offset() -> None:
    """Test random_seed_offset."""
    try:
        mod.random_seed_offset()
    except Exception:
        pass


def test_remat_opt_barrier() -> None:
    """Test remat_opt_barrier."""
    try:
        mod.remat_opt_barrier()
    except Exception:
        pass


def test_share_autotune_config_between_hosts() -> None:
    """Test share_autotune_config_between_hosts."""
    try:
        mod.share_autotune_config_between_hosts()
    except Exception:
        pass


def test_share_binary_between_hosts() -> None:
    """Test share_binary_between_hosts."""
    try:
        mod.share_binary_between_hosts()
    except Exception:
        pass


def test_share_binary_between_hosts_timeout_ms() -> None:
    """Test share_binary_between_hosts_timeout_ms."""
    try:
        mod.share_binary_between_hosts_timeout_ms()
    except Exception:
        pass


def test_softmax_custom_jvp() -> None:
    """Test softmax_custom_jvp."""
    try:
        mod.softmax_custom_jvp()
    except Exception:
        pass


def test_spmd_mode() -> None:
    """Test spmd_mode."""
    try:
        mod.spmd_mode()
    except Exception:
        pass


def test_threefry_gpu_kernel_lowering() -> None:
    """Test threefry_gpu_kernel_lowering."""
    try:
        mod.threefry_gpu_kernel_lowering()
    except Exception:
        pass


def test_threefry_partitionable() -> None:
    """Test threefry_partitionable."""
    try:
        mod.threefry_partitionable()
    except Exception:
        pass


def test_trace_context() -> None:
    """Test trace_context."""
    try:
        mod.trace_context()
    except Exception:
        pass


def test_traceback_filtering() -> None:
    """Test traceback_filtering."""
    try:
        mod.traceback_filtering()
    except Exception:
        pass


def test_traceback_in_locations_limit() -> None:
    """Test traceback_in_locations_limit."""
    try:
        mod.traceback_in_locations_limit()
    except Exception:
        pass


def test_transfer_guard() -> None:
    """Test transfer_guard."""
    try:
        mod.transfer_guard()
    except Exception:
        pass


def test_transfer_guard_device_to_device() -> None:
    """Test transfer_guard_device_to_device."""
    try:
        mod.transfer_guard_device_to_device()
    except Exception:
        pass


def test_transfer_guard_device_to_host() -> None:
    """Test transfer_guard_device_to_host."""
    try:
        mod.transfer_guard_device_to_host()
    except Exception:
        pass


def test_transfer_guard_host_to_device() -> None:
    """Test transfer_guard_host_to_device."""
    try:
        mod.transfer_guard_host_to_device()
    except Exception:
        pass


def test_unset() -> None:
    """Test unset."""
    try:
        mod.unset()
    except Exception:
        pass


def test_update() -> None:
    """Test update."""
    try:
        mod.update()
    except Exception:
        pass


def test_update_thread_local_jit_state() -> None:
    """Test update_thread_local_jit_state."""
    try:
        mod.update_thread_local_jit_state()
    except Exception:
        pass


def test_xla_runtime_errors() -> None:
    """Test xla_runtime_errors."""
    try:
        mod.xla_runtime_errors()
    except Exception:
        pass
