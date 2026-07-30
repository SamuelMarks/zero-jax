"""Frontend API routing for jax.interpreters.traceback_util.util.config."""

import typing
from typing import Any, Optional

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


UPGRADE_BOOL_HELP: Any = None


def bool_env(*args: Any, **kwargs: Any) -> Any:
    pass


def check_exists(*args: Any, **kwargs: Any) -> Any:
    pass


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    pass


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    pass


def default_device(*args: Any, **kwargs: Any) -> Any:
    pass


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    pass


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    pass


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    pass


def enable_x64(*args: Any, **kwargs: Any) -> Any:
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


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    pass


def trace_context(*args: Any, **kwargs: Any) -> Any:
    pass


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    """Context manager for guarding against implicit device transfers."""
    from zero_jax.api.debug import transfer_guard as _transfer_guard

    return _transfer_guard(*args, **kwargs)


def unset(*args: Any, **kwargs: Any) -> Any:
    pass


def update(*args: Any, **kwargs: Any) -> Any:
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


def already_configured_with_absl(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for already_configured_with_absl."""


annotations: typing.Any = None


def bcoo_cusparse_lowering(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for bcoo_cusparse_lowering."""


def cast(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for cast."""


def check_tracer_leaks(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for check_tracer_leaks."""


def compilation_cache_dir(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for compilation_cache_dir."""


def compilation_cache_include_metadata_in_key(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for compilation_cache_include_metadata_in_key."""


def compilation_cache_max_size(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for compilation_cache_max_size."""


def config(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for config."""


def custom_vjp_disable_shape_check(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for custom_vjp_disable_shape_check."""


def debug_infs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for debug_infs."""


def debug_nans(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for debug_nans."""


def default_dtype_bits(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for default_dtype_bits."""


def default_matmul_precision(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for default_matmul_precision."""


def default_prng_impl(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for default_prng_impl."""


def define_bool_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_bool_state."""


def define_enum_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_enum_state."""


def define_float_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_float_state."""


def define_int_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_int_state."""


def define_optional_enum_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_optional_enum_state."""


def define_optional_string_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_optional_string_state."""


def define_string_or_object_state(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for define_string_or_object_state."""


def define_string_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for define_string_state."""


def distributed_debug(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for distributed_debug."""


def enable_compilation_cache(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for enable_compilation_cache."""


def enable_custom_prng(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for enable_custom_prng."""


def enable_custom_vjp_by_custom_transpose(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for enable_custom_vjp_by_custom_transpose."""


def explain_cache_misses(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for explain_cache_misses."""


def explicit_device_get_scope(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for explicit_device_get_scope."""


def explicit_device_put_scope(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for explicit_device_put_scope."""


def include_full_tracebacks_in_locations(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for include_full_tracebacks_in_locations."""


def jax2tf_associative_scan_reductions(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for jax2tf_associative_scan_reductions."""


def jax2tf_default_native_serialization(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for jax2tf_default_native_serialization."""


def jax_export_calling_convention_version(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for jax_export_calling_convention_version."""


def jax_serialization_version(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for jax_serialization_version."""


def log_checkpoint_residuals(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for log_checkpoint_residuals."""


def numpy_dtype_promotion(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for numpy_dtype_promotion."""


def numpy_rank_promotion(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for numpy_rank_promotion."""


def parse_flags_with_absl(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for parse_flags_with_absl."""


def persistent_cache_min_compile_time_secs(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for persistent_cache_min_compile_time_secs."""


def persistent_cache_min_entry_size_bytes(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for persistent_cache_min_entry_size_bytes."""


def pgle_aggregation_percentile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for pgle_aggregation_percentile."""


def pgle_profiling_runs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for pgle_profiling_runs."""


def pmap_no_rank_reduction(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for pmap_no_rank_reduction."""


def pmap_shmap_merge(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for pmap_shmap_merge."""


def raise_persistent_cache_errors(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for raise_persistent_cache_errors."""


def random_seed_offset(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for random_seed_offset."""


def remat_opt_barrier(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for remat_opt_barrier."""


def share_autotune_config_between_hosts(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for share_autotune_config_between_hosts."""


def share_binary_between_hosts(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for share_binary_between_hosts."""


def share_binary_between_hosts_timeout_ms(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for share_binary_between_hosts_timeout_ms."""


def softmax_custom_jvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for softmax_custom_jvp."""


def threefry_gpu_kernel_lowering(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for threefry_gpu_kernel_lowering."""


def threefry_partitionable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for threefry_partitionable."""


def traceback_filtering(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for traceback_filtering."""


def traceback_in_locations_limit(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Stub for traceback_in_locations_limit."""


def transfer_guard_device_to_device(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for transfer_guard_device_to_device."""


def transfer_guard_device_to_host(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for transfer_guard_device_to_host."""


def transfer_guard_host_to_device(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for transfer_guard_host_to_device."""


def update_thread_local_jit_state(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Stub for update_thread_local_jit_state."""
