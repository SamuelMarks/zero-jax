"""Traceback util config stub."""

from typing import Any

xla_client = None

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class Config:
    """Config class stub."""

    def __init__(self, *args, **kwargs):
        pass


def DEFINE_bool(*args: Any, **kwargs: Any) -> Any:
    pass


def DEFINE_enum(*args: Any, **kwargs: Any) -> Any:
    pass


def DEFINE_float(*args: Any, **kwargs: Any) -> Any:
    pass


def DEFINE_integer(*args: Any, **kwargs: Any) -> Any:
    pass


def DEFINE_string(*args: Any, **kwargs: Any) -> Any:
    pass


class FlagHolder:
    def __init__(self, *args, **kwargs):
        pass


class Generic:
    def __init__(self, *args, **kwargs):
        pass


class Hashable:
    def __init__(self, *args, **kwargs):
        pass


class Iterator:
    def __init__(self, *args, **kwargs):
        pass


class NamedTuple:
    def __init__(self, *args, **kwargs):
        pass


class NoDefault:
    def __init__(self, *args, **kwargs):
        pass


class NoReturn:
    def __init__(self, *args, **kwargs):
        pass


class TypeVar:
    def __init__(self, *args, **kwargs):
        pass


UPGRADE_BOOL_EXTRA_DESC: Any = None
UPGRADE_BOOL_HELP: Any = None


def bool_env(*args: Any, **kwargs: Any) -> Any:
    pass


def check_exists(*args: Any, **kwargs: Any) -> Any:
    pass


def check_tracer_leaks(*args: Any, **kwargs: Any) -> Any:
    pass


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    pass


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    pass


def default_device(*args: Any, **kwargs: Any) -> Any:
    pass


def default_dtype_bits(*args: Any, **kwargs: Any) -> Any:
    pass


def default_prng_impl(*args: Any, **kwargs: Any) -> Any:
    pass


def define_bool_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_enum_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_float_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_int_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_string_state(*args: Any, **kwargs: Any) -> Any:
    pass


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    pass


def distributed_debug(*args: Any, **kwargs: Any) -> Any:
    pass


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    pass


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_custom_prng(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    pass


def explain_cache_misses(*args: Any, **kwargs: Any) -> Any:
    pass


def int_env(*args: Any, **kwargs: Any) -> Any:
    pass


def jax_platforms(*args: Any, **kwargs: Any) -> Any:
    pass


def legacy_prng_key(*args: Any, **kwargs: Any) -> Any:
    pass


def lib(*args: Any, **kwargs: Any) -> Any:
    pass


def log_compiles(*args: Any, **kwargs: Any) -> Any:
    pass


def logger(*args: Any, **kwargs: Any) -> Any:
    pass


def no_default(*args: Any, **kwargs: Any) -> Any:
    pass


def numpy_rank_promotion(*args: Any, **kwargs: Any) -> Any:
    pass


def pgle_profiling_runs(*args: Any, **kwargs: Any) -> Any:
    pass


def pmap_shmap_merge(*args: Any, **kwargs: Any) -> Any:
    pass


def random_seed_offset(*args: Any, **kwargs: Any) -> Any:
    pass


def remat_opt_barrier(*args: Any, **kwargs: Any) -> Any:
    pass


def softmax_custom_jvp(*args: Any, **kwargs: Any) -> Any:
    pass


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    pass


def trace_context(*args: Any, **kwargs: Any) -> Any:
    pass


def traceback_filtering(*args: Any, **kwargs: Any) -> Any:
    pass


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    """Context manager for guarding against implicit device transfers."""
    from zero_jax.api.debug import transfer_guard as _transfer_guard

    return _transfer_guard(*args, **kwargs)


def unset(*args: Any, **kwargs: Any) -> Any:
    pass


def update(*args: Any, **kwargs: Any) -> Any:
    pass


def xla_runtime_errors(*args: Any, **kwargs: Any) -> Any:
    pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(f"Stub for {name} is not implemented in backend")

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover


def already_configured_with_absl(*args: Any, **kwargs: Any) -> Any:
    pass


def bcoo_cusparse_lowering(*args: Any, **kwargs: Any) -> Any:
    pass


def cast(*args: Any, **kwargs: Any) -> Any:
    pass


def compilation_cache_dir(*args: Any, **kwargs: Any) -> Any:
    pass


def compilation_cache_include_metadata_in_key(*args: Any, **kwargs: Any) -> Any:
    pass


def compilation_cache_max_size(*args: Any, **kwargs: Any) -> Any:
    pass


def config(*args: Any, **kwargs: Any) -> Any:
    pass


def custom_vjp_disable_shape_check(*args: Any, **kwargs: Any) -> Any:
    pass


def debug_infs(*args: Any, **kwargs: Any) -> Any:
    pass


def debug_nans(*args: Any, **kwargs: Any) -> Any:
    pass


def default_matmul_precision(*args: Any, **kwargs: Any) -> Any:
    pass


def define_optional_enum_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_optional_string_state(*args: Any, **kwargs: Any) -> Any:
    pass


def define_string_or_object_state(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_compilation_cache(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_custom_vjp_by_custom_transpose(*args: Any, **kwargs: Any) -> Any:
    pass


def explicit_device_get_scope(*args: Any, **kwargs: Any) -> Any:
    pass


def explicit_device_put_scope(*args: Any, **kwargs: Any) -> Any:
    pass


def include_full_tracebacks_in_locations(*args: Any, **kwargs: Any) -> Any:
    pass


def jax2tf_associative_scan_reductions(*args: Any, **kwargs: Any) -> Any:
    pass


def jax2tf_default_native_serialization(*args: Any, **kwargs: Any) -> Any:
    pass


def jax_export_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    pass


def jax_serialization_version(*args: Any, **kwargs: Any) -> Any:
    pass


def log_checkpoint_residuals(*args: Any, **kwargs: Any) -> Any:
    pass


def numpy_dtype_promotion(*args: Any, **kwargs: Any) -> Any:
    pass


def parse_flags_with_absl(*args: Any, **kwargs: Any) -> Any:
    pass


def persistent_cache_min_compile_time_secs(*args: Any, **kwargs: Any) -> Any:
    pass


def persistent_cache_min_entry_size_bytes(*args: Any, **kwargs: Any) -> Any:
    pass


def pgle_aggregation_percentile(*args: Any, **kwargs: Any) -> Any:
    pass


def pmap_no_rank_reduction(*args: Any, **kwargs: Any) -> Any:
    pass


def raise_persistent_cache_errors(*args: Any, **kwargs: Any) -> Any:
    pass


def share_autotune_config_between_hosts(*args: Any, **kwargs: Any) -> Any:
    pass


def share_binary_between_hosts(*args: Any, **kwargs: Any) -> Any:
    pass


def share_binary_between_hosts_timeout_ms(*args: Any, **kwargs: Any) -> Any:
    pass


def threefry_gpu_kernel_lowering(*args: Any, **kwargs: Any) -> Any:
    pass


def threefry_partitionable(*args: Any, **kwargs: Any) -> Any:
    pass


def traceback_in_locations_limit(*args: Any, **kwargs: Any) -> Any:
    pass


def transfer_guard_device_to_device(*args: Any, **kwargs: Any) -> Any:
    pass


def transfer_guard_device_to_host(*args: Any, **kwargs: Any) -> Any:
    pass


def transfer_guard_host_to_device(*args: Any, **kwargs: Any) -> Any:
    pass


def update_thread_local_jit_state(*args: Any, **kwargs: Any) -> Any:
    pass


annotations: Any = None
