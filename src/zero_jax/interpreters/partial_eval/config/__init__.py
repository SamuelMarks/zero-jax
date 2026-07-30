"""Config stub."""

from typing import Any

from . import logging_config

jax_jit = True


import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class Config:
    """Stub Config class."""

    def __init__(self, *args, **kwargs):
        pass


def DEFINE_bool(*args: Any, **kwargs: Any) -> Any:
    return None


def DEFINE_enum(*args: Any, **kwargs: Any) -> Any:
    return None


def DEFINE_float(*args: Any, **kwargs: Any) -> Any:
    return None


def DEFINE_integer(*args: Any, **kwargs: Any) -> Any:
    return None


def DEFINE_string(*args: Any, **kwargs: Any) -> Any:
    return None


def FlagHolder(*args: Any, **kwargs: Any) -> Any:
    return None


def Generic(*args: Any, **kwargs: Any) -> Any:
    return None


def Hashable(*args: Any, **kwargs: Any) -> Any:
    return None


def Iterator(*args: Any, **kwargs: Any) -> Any:
    return None


def NamedTuple(*args: Any, **kwargs: Any) -> Any:
    return None


def NoDefault(*args: Any, **kwargs: Any) -> Any:
    return None


def NoReturn(*args: Any, **kwargs: Any) -> Any:
    return None


def TypeVar(*args: Any, **kwargs: Any) -> Any:
    return None


UPGRADE_BOOL_EXTRA_DESC = ""
UPGRADE_BOOL_HELP = ""


def bcoo_cusparse_lowering(*args: Any, **kwargs: Any) -> Any:
    return None


def bool_env(*args: Any, **kwargs: Any) -> Any:
    return None


def check_exists(*args: Any, **kwargs: Any) -> Any:
    return None


def check_tracer_leaks(*args: Any, **kwargs: Any) -> Any:
    return None


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    return None


def compilation_cache_dir(*args: Any, **kwargs: Any) -> Any:
    return None


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    return None


def default_device(*args: Any, **kwargs: Any) -> Any:
    return None


def default_dtype_bits(*args: Any, **kwargs: Any) -> Any:
    return None


def default_prng_impl(*args: Any, **kwargs: Any) -> Any:
    return None


def define_bool_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_enum_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_float_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_int_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_string_state(*args: Any, **kwargs: Any) -> Any:
    return None


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    return None


def distributed_debug(*args: Any, **kwargs: Any) -> Any:
    return None


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    return None


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_custom_prng(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    return None


def explain_cache_misses(*args: Any, **kwargs: Any) -> Any:
    return None


def int_env(*args: Any, **kwargs: Any) -> Any:
    return None


def jax_platforms(*args: Any, **kwargs: Any) -> Any:
    return None


def legacy_prng_key(*args: Any, **kwargs: Any) -> Any:
    return None


def lib(*args: Any, **kwargs: Any) -> Any:
    return None


def log_compiles(*args: Any, **kwargs: Any) -> Any:
    return None


def logger(*args: Any, **kwargs: Any) -> Any:
    return None


def no_default(*args: Any, **kwargs: Any) -> Any:
    return None


def numpy_dtype_promotion(*args: Any, **kwargs: Any) -> Any:
    return None


def numpy_rank_promotion(*args: Any, **kwargs: Any) -> Any:
    return None


def parse_flags_with_absl(*args: Any, **kwargs: Any) -> Any:
    return None


def pgle_profiling_runs(*args: Any, **kwargs: Any) -> Any:
    return None


def pmap_no_rank_reduction(*args: Any, **kwargs: Any) -> Any:
    return None


def pmap_shmap_merge(*args: Any, **kwargs: Any) -> Any:
    return None


def random_seed_offset(*args: Any, **kwargs: Any) -> Any:
    return None


def remat_opt_barrier(*args: Any, **kwargs: Any) -> Any:
    return None


def softmax_custom_jvp(*args: Any, **kwargs: Any) -> Any:
    return None


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    return None


def threefry_partitionable(*args: Any, **kwargs: Any) -> Any:
    return None


def trace_context(*args: Any, **kwargs: Any) -> Any:
    return None


def traceback_filtering(*args: Any, **kwargs: Any) -> Any:
    return None


def unset(*args: Any, **kwargs: Any) -> Any:
    return None


def update(*args: Any, **kwargs: Any) -> Any:
    return None


def xla_runtime_errors(*args: Any, **kwargs: Any) -> Any:
    return None


def already_configured_with_absl(*args: Any, **kwargs: Any) -> Any:
    return None


def cast(*args: Any, **kwargs: Any) -> Any:
    return None


def compilation_cache_include_metadata_in_key(*args: Any, **kwargs: Any) -> Any:
    return None


def compilation_cache_max_size(*args: Any, **kwargs: Any) -> Any:
    return None


def config(*args: Any, **kwargs: Any) -> Any:
    return None


def custom_vjp_disable_shape_check(*args: Any, **kwargs: Any) -> Any:
    return None


def debug_infs(*args: Any, **kwargs: Any) -> Any:
    return None


def debug_nans(*args: Any, **kwargs: Any) -> Any:
    return None


def default_matmul_precision(*args: Any, **kwargs: Any) -> Any:
    return None


def define_optional_enum_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_optional_string_state(*args: Any, **kwargs: Any) -> Any:
    return None


def define_string_or_object_state(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_compilation_cache(*args: Any, **kwargs: Any) -> Any:
    return None


def enable_custom_vjp_by_custom_transpose(*args: Any, **kwargs: Any) -> Any:
    return None


def explicit_device_get_scope(*args: Any, **kwargs: Any) -> Any:
    return None


def explicit_device_put_scope(*args: Any, **kwargs: Any) -> Any:
    return None


def hlo_source_file_canonicalization_regex(*args: Any, **kwargs: Any) -> Any:
    return None


def include_full_tracebacks_in_locations(*args: Any, **kwargs: Any) -> Any:
    return None


def jax2tf_associative_scan_reductions(*args: Any, **kwargs: Any) -> Any:
    return None


def jax2tf_default_native_serialization(*args: Any, **kwargs: Any) -> Any:
    return None


def jax_export_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    return None


def jax_pjrt_client_create_options(*args: Any, **kwargs: Any) -> Any:
    return None


def jax_serialization_version(*args: Any, **kwargs: Any) -> Any:
    return None


def jax_xla_profile_version(*args: Any, **kwargs: Any) -> Any:
    return None


def log_checkpoint_residuals(*args: Any, **kwargs: Any) -> Any:
    return None


def persistent_cache_min_compile_time_secs(*args: Any, **kwargs: Any) -> Any:
    return None


def persistent_cache_min_entry_size_bytes(*args: Any, **kwargs: Any) -> Any:
    return None


def pgle_aggregation_percentile(*args: Any, **kwargs: Any) -> Any:
    return None


def raise_persistent_cache_errors(*args: Any, **kwargs: Any) -> Any:
    return None


def share_autotune_config_between_hosts(*args: Any, **kwargs: Any) -> Any:
    return None


def share_binary_between_hosts(*args: Any, **kwargs: Any) -> Any:
    return None


def share_binary_between_hosts_timeout_ms(*args: Any, **kwargs: Any) -> Any:
    return None


def threefry_gpu_kernel_lowering(*args: Any, **kwargs: Any) -> Any:
    return None


def traceback_in_locations_limit(*args: Any, **kwargs: Any) -> Any:
    return None


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    return None  # pragma: no cover


def transfer_guard_device_to_device(*args: Any, **kwargs: Any) -> Any:
    return None


def transfer_guard_device_to_host(*args: Any, **kwargs: Any) -> Any:
    return None


def transfer_guard_host_to_device(*args: Any, **kwargs: Any) -> Any:
    return None


def update_thread_local_jit_state(*args: Any, **kwargs: Any) -> Any:
    return None


def Callable(*args: Any, **kwargs: Any) -> Any:
    return None


annotations: Any = None
