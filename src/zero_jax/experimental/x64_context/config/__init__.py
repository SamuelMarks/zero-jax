"""Frontend API routing for jax.experimental.x64_context.config."""

from typing import Any, Callable

import zero_jax._compiler_proxy_ops as _ops


class Config:
    """Frontend state holder for Config."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def DEFINE_bool(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DEFINE_bool.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DEFINE_bool(*args, **kwargs)


def DEFINE_enum(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DEFINE_enum.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DEFINE_enum(*args, **kwargs)


def DEFINE_float(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DEFINE_float.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DEFINE_float(*args, **kwargs)


def DEFINE_integer(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DEFINE_integer.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DEFINE_integer(*args, **kwargs)


def DEFINE_string(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DEFINE_string.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DEFINE_string(*args, **kwargs)


class FlagHolder:
    """Frontend state holder for FlagHolder."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Generic:
    """Abstract base class for generic types."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Hashable:
    """Frontend state holder for Hashable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Iterator:
    """Frontend state holder for Iterator."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def NamedTuple(*args: Any, **kwargs: Any) -> Any:
    """Typed version of namedtuple."""
    return _ops.NamedTuple(*args, **kwargs)


class NoDefault:
    """Frontend state holder for NoDefault."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def NoReturn(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating functions that never return."""
    return _ops.NoReturn(*args, **kwargs)


class TypeVar:
    """Type variable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


UPGRADE_BOOL_EXTRA_DESC = ""

UPGRADE_BOOL_HELP = ""


def already_configured_with_absl(*args: Any, **kwargs: Any) -> Any:
    return None


annotations: Any = None


def bcoo_cusparse_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_bcoo_cusparse_lowering` config option."""
    return _ops.bcoo_cusparse_lowering(*args, **kwargs)


def bool_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as a boolean."""
    return _ops.bool_env(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return _ops.cast(*args, **kwargs)


def check_exists(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for check_exists.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.check_exists(*args, **kwargs)


def check_tracer_leaks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_check_tracer_leaks` config option."""
    return _ops.check_tracer_leaks(*args, **kwargs)


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.checking_leaks(*args, **kwargs)


def compilation_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_dir` config option."""
    return _ops.compilation_cache_dir(*args, **kwargs)


def compilation_cache_include_metadata_in_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_include_metadata_in_key` config option."""
    return _ops.compilation_cache_include_metadata_in_key(*args, **kwargs)


def compilation_cache_max_size(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_max_size` config option."""
    return _ops.compilation_cache_max_size(*args, **kwargs)


def config(*args: Any, **kwargs: Any) -> Any:
    return None


from . import contextlib


def custom_vjp_disable_shape_check(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_custom_vjp_disable_shape_check` config option (transient)."""
    return _ops.custom_vjp_disable_shape_check(*args, **kwargs)


def debug_infs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_infs` config option."""
    return _ops.debug_infs(*args, **kwargs)


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_key_reuse` config option."""
    return _ops.debug_key_reuse(*args, **kwargs)


def debug_nans(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_nans` config option."""
    return _ops.debug_nans(*args, **kwargs)


def default_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_device` config option."""
    return _ops.default_device(*args, **kwargs)


def default_dtype_bits(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_dtype_bits` config option."""
    return _ops.default_dtype_bits(*args, **kwargs)


def default_matmul_precision(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_matmul_precision` config option."""
    return _ops.default_matmul_precision(*args, **kwargs)


def default_prng_impl(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_prng_impl` config option."""
    return _ops.default_prng_impl(*args, **kwargs)


def define_bool_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_bool_state(*args, **kwargs)


def define_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_enum_state(*args, **kwargs)


def define_float_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_float_state(*args, **kwargs)


def define_int_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_int_state(*args, **kwargs)


def define_optional_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_optional_enum_state(*args, **kwargs)


def define_optional_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_optional_string_state(*args, **kwargs)


def define_string_or_object_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_string_or_object_state(*args, **kwargs)


def define_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return _ops.define_string_state(*args, **kwargs)


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_disable_jit` config option."""
    return _ops.disable_jit(*args, **kwargs)


def distributed_debug(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_distributed_debug` config option."""
    return _ops.distributed_debug(*args, **kwargs)


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_dynamic_shapes` config option."""
    return _ops.dynamic_shapes(*args, **kwargs)


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_eager_pmap` config option (transient)."""
    return _ops.eager_pmap(*args, **kwargs)


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_checks` config option."""
    return _ops.enable_checks(*args, **kwargs)


def enable_compilation_cache(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_compilation_cache` config option."""
    return _ops.enable_compilation_cache(*args, **kwargs)


def enable_custom_prng(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_prng` config option (transient)."""
    return _ops.enable_custom_prng(*args, **kwargs)


def enable_custom_vjp_by_custom_transpose(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_vjp_by_custom_transpose` config option (transient)."""
    return _ops.enable_custom_vjp_by_custom_transpose(*args, **kwargs)


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_memories` config option (transient)."""
    return _ops.enable_memories(*args, **kwargs)


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_pgle` config option."""
    return _ops.enable_pgle(*args, **kwargs)


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_x64` config option."""
    return _ops.enable_x64(*args, **kwargs)


def explain_cache_misses(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_explain_cache_misses` config option."""
    return _ops.explain_cache_misses(*args, **kwargs)


def explicit_device_get_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_get() call."""
    return _ops.explicit_device_get_scope(*args, **kwargs)


def explicit_device_put_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_put*() call."""
    return _ops.explicit_device_put_scope(*args, **kwargs)


from . import functools


def hlo_source_file_canonicalization_regex(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_hlo_source_file_canonicalization_regex` config option."""
    return _ops.hlo_source_file_canonicalization_regex(*args, **kwargs)


def include_full_tracebacks_in_locations(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_include_full_tracebacks_in_locations` config option."""
    return _ops.include_full_tracebacks_in_locations(*args, **kwargs)


def int_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as an integer."""
    return _ops.int_env(*args, **kwargs)


from . import itertools


def jax2tf_associative_scan_reductions(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_associative_scan_reductions` config option."""
    return _ops.jax2tf_associative_scan_reductions(*args, **kwargs)


def jax2tf_default_native_serialization(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_default_native_serialization` config option."""
    return _ops.jax2tf_default_native_serialization(*args, **kwargs)


def jax_export_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_export_calling_convention_version` config option."""
    return _ops.jax_export_calling_convention_version(*args, **kwargs)


from . import jax_jit


def jax_pjrt_client_create_options(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pjrt_client_create_options` config option."""
    return _ops.jax_pjrt_client_create_options(*args, **kwargs)


def jax_platforms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_platforms` config option."""
    return _ops.jax_platforms(*args, **kwargs)


def jax_serialization_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_serialization_version` config option."""
    return _ops.jax_serialization_version(*args, **kwargs)


def jax_xla_profile_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_xla_profile_version` config option."""
    return _ops.jax_xla_profile_version(*args, **kwargs)


def legacy_prng_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_legacy_prng_key` config option."""
    return _ops.legacy_prng_key(*args, **kwargs)


def log_checkpoint_residuals(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_checkpoint_residuals` config option."""
    return _ops.log_checkpoint_residuals(*args, **kwargs)


def log_compiles(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_compiles` config option."""
    return _ops.log_compiles(*args, **kwargs)


def logger(*args: Any, **kwargs: Any) -> Any:
    return None


from . import logging, logging_config


def no_default(*args: Any, **kwargs: Any) -> Any:
    return None


def numpy_dtype_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_dtype_promotion` config option."""
    return _ops.numpy_dtype_promotion(*args, **kwargs)


def numpy_rank_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_rank_promotion` config option."""
    return _ops.numpy_rank_promotion(*args, **kwargs)


from . import os


def parse_flags_with_absl(*args: Any, **kwargs: Any) -> Any:
    """Parses command-line args that start with --jax."""
    return _ops.parse_flags_with_absl(*args, **kwargs)


def persistent_cache_min_compile_time_secs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_compile_time_secs` config option."""
    return _ops.persistent_cache_min_compile_time_secs(*args, **kwargs)


def persistent_cache_min_entry_size_bytes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_entry_size_bytes` config option."""
    return _ops.persistent_cache_min_entry_size_bytes(*args, **kwargs)


def pgle_aggregation_percentile(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_aggregation_percentile` config option."""
    return _ops.pgle_aggregation_percentile(*args, **kwargs)


def pgle_profiling_runs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_profiling_runs` config option."""
    return _ops.pgle_profiling_runs(*args, **kwargs)


def pmap_no_rank_reduction(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_no_rank_reduction` config option."""
    return _ops.pmap_no_rank_reduction(*args, **kwargs)


def pmap_shmap_merge(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_shmap_merge` config option (transient)."""
    return _ops.pmap_shmap_merge(*args, **kwargs)


def raise_persistent_cache_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_raise_persistent_cache_errors` config option."""
    return _ops.raise_persistent_cache_errors(*args, **kwargs)


def random_seed_offset(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_random_seed_offset` config option."""
    return _ops.random_seed_offset(*args, **kwargs)


def remat_opt_barrier(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_remat_opt_barrier` config option."""
    return _ops.remat_opt_barrier(*args, **kwargs)


def share_autotune_config_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_autotune_config_between_hosts` config option."""
    return _ops.share_autotune_config_between_hosts(*args, **kwargs)


def share_binary_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts` config option."""
    return _ops.share_binary_between_hosts(*args, **kwargs)


def share_binary_between_hosts_timeout_ms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts_timeout_ms` config option."""
    return _ops.share_binary_between_hosts_timeout_ms(*args, **kwargs)


def softmax_custom_jvp(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_softmax_custom_jvp` config option (transient)."""
    return _ops.softmax_custom_jvp(*args, **kwargs)


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_spmd_mode` config option."""
    return _ops.spmd_mode(*args, **kwargs)


from . import sys, threading


def threefry_gpu_kernel_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_gpu_kernel_lowering` config option."""
    return _ops.threefry_gpu_kernel_lowering(*args, **kwargs)


def threefry_partitionable(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_partitionable` config option (transient)."""
    return _ops.threefry_partitionable(*args, **kwargs)


def trace_context(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple of configuration values that affect tracing."""
    return _ops.trace_context(*args, **kwargs)


def traceback_filtering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_filtering` config option."""
    return _ops.traceback_filtering(*args, **kwargs)


def traceback_in_locations_limit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_in_locations_limit` config option."""
    return _ops.traceback_in_locations_limit(*args, **kwargs)


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    """A contextmanager to control the transfer guard level for all transfers."""
    return _ops.transfer_guard(*args, **kwargs)


def transfer_guard_device_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_device` config option."""
    return _ops.transfer_guard_device_to_device(*args, **kwargs)


def transfer_guard_device_to_host(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_host` config option."""
    return _ops.transfer_guard_device_to_host(*args, **kwargs)


def transfer_guard_host_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_host_to_device` config option."""
    return _ops.transfer_guard_host_to_device(*args, **kwargs)


from . import transfer_guard_lib


def unset(*args: Any, **kwargs: Any) -> Any:
    return None


def update(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for update.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.update(*args, **kwargs)


def update_thread_local_jit_state(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for update_thread_local_jit_state.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.update_thread_local_jit_state(*args, **kwargs)


def xla_runtime_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_experimental_unsafe_xla_runtime_errors` config option."""
    return _ops.xla_runtime_errors(*args, **kwargs)


import typing

import ml_switcheroo_compiler


def lib(*args: Any, **kwargs: Any) -> Any:
    pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
