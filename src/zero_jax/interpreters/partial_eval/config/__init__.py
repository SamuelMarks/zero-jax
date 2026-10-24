"""Config stub."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue

from . import logging_config

jax_jit = True


import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class Config:
    """Stub Config class."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


def DEFINE_bool(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``DEFINE_bool``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def DEFINE_enum(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``DEFINE_enum``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def DEFINE_float(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``DEFINE_float``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def DEFINE_integer(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``DEFINE_integer``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def DEFINE_string(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``DEFINE_string``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def FlagHolder(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``FlagHolder``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def Generic(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``Generic``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def Hashable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``Hashable``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def Iterator(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``Iterator``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def NamedTuple(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``NamedTuple``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def NoDefault(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``NoDefault``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def NoReturn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``NoReturn``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def TypeVar(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``TypeVar``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


UPGRADE_BOOL_EXTRA_DESC = ""
UPGRADE_BOOL_HELP = ""


def bcoo_cusparse_lowering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``bcoo_cusparse_lowering``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def bool_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``bool_env``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def check_exists(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``check_exists``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def check_tracer_leaks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``check_tracer_leaks``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def checking_leaks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``checking_leaks``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def compilation_cache_dir(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``compilation_cache_dir``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def debug_key_reuse(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``debug_key_reuse``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def default_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``default_device``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def default_dtype_bits(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``default_dtype_bits``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def default_prng_impl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``default_prng_impl``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_bool_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_bool_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_enum_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_enum_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_float_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_float_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_int_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_int_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_string_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_string_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def disable_jit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``disable_jit``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def distributed_debug(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``distributed_debug``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def dynamic_shapes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``dynamic_shapes``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def eager_pmap(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``eager_pmap``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_checks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_checks``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_custom_prng(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_custom_prng``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_memories(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_memories``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_pgle(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_pgle``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_x64(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_x64``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def explain_cache_misses(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``explain_cache_misses``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def int_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``int_env``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax_platforms(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax_platforms``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def legacy_prng_key(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``legacy_prng_key``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def lib(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``lib``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def log_compiles(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``log_compiles``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def logger(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``logger``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def no_default(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``no_default``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def numpy_dtype_promotion(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``numpy_dtype_promotion``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def numpy_rank_promotion(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``numpy_rank_promotion``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def parse_flags_with_absl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``parse_flags_with_absl``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def pgle_profiling_runs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pgle_profiling_runs``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def pmap_no_rank_reduction(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pmap_no_rank_reduction``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def pmap_shmap_merge(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pmap_shmap_merge``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def random_seed_offset(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``random_seed_offset``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def remat_opt_barrier(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``remat_opt_barrier``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def softmax_custom_jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``softmax_custom_jvp``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def spmd_mode(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``spmd_mode``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def threefry_partitionable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``threefry_partitionable``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def trace_context(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``trace_context``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def traceback_filtering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``traceback_filtering``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def unset(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``unset``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def update(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``update``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def xla_runtime_errors(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``xla_runtime_errors``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def already_configured_with_absl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``already_configured_with_absl``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def cast(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``cast``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def compilation_cache_include_metadata_in_key(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``compilation_cache_include_metadata_in_key``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def compilation_cache_max_size(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``compilation_cache_max_size``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def config(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``config``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def custom_vjp_disable_shape_check(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``custom_vjp_disable_shape_check``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def debug_infs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``debug_infs``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def debug_nans(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``debug_nans``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def default_matmul_precision(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``default_matmul_precision``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_optional_enum_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_optional_enum_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_optional_string_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_optional_string_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def define_string_or_object_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``define_string_or_object_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_compilation_cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_compilation_cache``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def enable_custom_vjp_by_custom_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``enable_custom_vjp_by_custom_transpose``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def explicit_device_get_scope(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``explicit_device_get_scope``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def explicit_device_put_scope(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``explicit_device_put_scope``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def hlo_source_file_canonicalization_regex(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``hlo_source_file_canonicalization_regex``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def include_full_tracebacks_in_locations(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``include_full_tracebacks_in_locations``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax2tf_associative_scan_reductions(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax2tf_associative_scan_reductions``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax2tf_default_native_serialization(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax2tf_default_native_serialization``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax_export_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax_export_calling_convention_version``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax_pjrt_client_create_options(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax_pjrt_client_create_options``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax_serialization_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax_serialization_version``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def jax_xla_profile_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jax_xla_profile_version``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def log_checkpoint_residuals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``log_checkpoint_residuals``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def persistent_cache_min_compile_time_secs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``persistent_cache_min_compile_time_secs``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def persistent_cache_min_entry_size_bytes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``persistent_cache_min_entry_size_bytes``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def pgle_aggregation_percentile(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pgle_aggregation_percentile``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def raise_persistent_cache_errors(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``raise_persistent_cache_errors``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def share_autotune_config_between_hosts(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``share_autotune_config_between_hosts``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def share_binary_between_hosts(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``share_binary_between_hosts``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def share_binary_between_hosts_timeout_ms(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``share_binary_between_hosts_timeout_ms``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def threefry_gpu_kernel_lowering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``threefry_gpu_kernel_lowering``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def traceback_in_locations_limit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``traceback_in_locations_limit``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def transfer_guard(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``transfer_guard``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def transfer_guard_device_to_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``transfer_guard_device_to_device``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def transfer_guard_device_to_host(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``transfer_guard_device_to_host``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def transfer_guard_host_to_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``transfer_guard_host_to_device``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def update_thread_local_jit_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``update_thread_local_jit_state``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def Callable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``Callable``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


annotations: ArrayLikeSpec = None
