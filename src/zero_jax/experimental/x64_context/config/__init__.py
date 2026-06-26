"""Frontend API routing for jax.experimental.x64_context.config."""

from typing import Any


class Config:
    """Mock implementation for Config."""

    pass


def DEFINE_bool(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_bool."""
    raise NotImplementedError("DEFINE_bool not yet implemented in zero-jax")


def DEFINE_enum(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_enum."""
    raise NotImplementedError("DEFINE_enum not yet implemented in zero-jax")


def DEFINE_float(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_float."""
    raise NotImplementedError("DEFINE_float not yet implemented in zero-jax")


def DEFINE_integer(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_integer."""
    raise NotImplementedError("DEFINE_integer not yet implemented in zero-jax")


def DEFINE_string(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_string."""
    raise NotImplementedError("DEFINE_string not yet implemented in zero-jax")


class FlagHolder:
    """Mock implementation for FlagHolder."""

    pass


class Generic:
    """Abstract base class for generic types."""

    pass


class Hashable:
    """Mock implementation for Hashable."""

    pass


class Iterator:
    """Mock implementation for Iterator."""

    pass


def NamedTuple(*args: Any, **kwargs: Any) -> Any:
    """Typed version of namedtuple."""
    raise NotImplementedError("NamedTuple not yet implemented in zero-jax")


class NoDefault:
    """Mock implementation for NoDefault."""

    pass


def NoReturn(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating functions that never return."""
    raise NotImplementedError("NoReturn not yet implemented in zero-jax")


class TypeVar:
    """Type variable."""

    pass


UPGRADE_BOOL_EXTRA_DESC: Any = None

UPGRADE_BOOL_HELP: Any = None

already_configured_with_absl: Any = None

annotations: Any = None


def bcoo_cusparse_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_bcoo_cusparse_lowering` config option."""
    raise NotImplementedError("bcoo_cusparse_lowering not yet implemented in zero-jax")


def bool_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as a boolean."""
    raise NotImplementedError("bool_env not yet implemented in zero-jax")


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    raise NotImplementedError("cast not yet implemented in zero-jax")


def check_exists(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_exists."""
    raise NotImplementedError("check_exists not yet implemented in zero-jax")


def check_tracer_leaks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_check_tracer_leaks` config option."""
    raise NotImplementedError("check_tracer_leaks not yet implemented in zero-jax")


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("checking_leaks not yet implemented in zero-jax")


def compilation_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_dir` config option."""
    raise NotImplementedError("compilation_cache_dir not yet implemented in zero-jax")


def compilation_cache_include_metadata_in_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_include_metadata_in_key` config option."""
    raise NotImplementedError(
        "compilation_cache_include_metadata_in_key not yet implemented in zero-jax"
    )


def compilation_cache_max_size(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_max_size` config option."""
    raise NotImplementedError(
        "compilation_cache_max_size not yet implemented in zero-jax"
    )


config: Any = None
from . import contextlib


def custom_vjp_disable_shape_check(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_custom_vjp_disable_shape_check` config option (transient)."""
    raise NotImplementedError(
        "custom_vjp_disable_shape_check not yet implemented in zero-jax"
    )


def debug_infs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_infs` config option."""
    raise NotImplementedError("debug_infs not yet implemented in zero-jax")


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_key_reuse` config option."""
    raise NotImplementedError("debug_key_reuse not yet implemented in zero-jax")


def debug_nans(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_nans` config option."""
    raise NotImplementedError("debug_nans not yet implemented in zero-jax")


def default_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_device` config option."""
    raise NotImplementedError("default_device not yet implemented in zero-jax")


def default_dtype_bits(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_dtype_bits` config option."""
    raise NotImplementedError("default_dtype_bits not yet implemented in zero-jax")


def default_matmul_precision(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_matmul_precision` config option."""
    raise NotImplementedError(
        "default_matmul_precision not yet implemented in zero-jax"
    )


def default_prng_impl(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_prng_impl` config option."""
    raise NotImplementedError("default_prng_impl not yet implemented in zero-jax")


def define_bool_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError("define_bool_state not yet implemented in zero-jax")


def define_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError("define_enum_state not yet implemented in zero-jax")


def define_float_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError("define_float_state not yet implemented in zero-jax")


def define_int_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError("define_int_state not yet implemented in zero-jax")


def define_optional_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError(
        "define_optional_enum_state not yet implemented in zero-jax"
    )


def define_optional_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError(
        "define_optional_string_state not yet implemented in zero-jax"
    )


def define_string_or_object_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError(
        "define_string_or_object_state not yet implemented in zero-jax"
    )


def define_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    raise NotImplementedError("define_string_state not yet implemented in zero-jax")


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_disable_jit` config option."""
    raise NotImplementedError("disable_jit not yet implemented in zero-jax")


def distributed_debug(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_distributed_debug` config option."""
    raise NotImplementedError("distributed_debug not yet implemented in zero-jax")


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_dynamic_shapes` config option."""
    raise NotImplementedError("dynamic_shapes not yet implemented in zero-jax")


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_eager_pmap` config option (transient)."""
    raise NotImplementedError("eager_pmap not yet implemented in zero-jax")


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_checks` config option."""
    raise NotImplementedError("enable_checks not yet implemented in zero-jax")


def enable_compilation_cache(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_compilation_cache` config option."""
    raise NotImplementedError(
        "enable_compilation_cache not yet implemented in zero-jax"
    )


def enable_custom_prng(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_prng` config option (transient)."""
    raise NotImplementedError("enable_custom_prng not yet implemented in zero-jax")


def enable_custom_vjp_by_custom_transpose(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_vjp_by_custom_transpose` config option (transient)."""
    raise NotImplementedError(
        "enable_custom_vjp_by_custom_transpose not yet implemented in zero-jax"
    )


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_memories` config option (transient)."""
    raise NotImplementedError("enable_memories not yet implemented in zero-jax")


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_pgle` config option."""
    raise NotImplementedError("enable_pgle not yet implemented in zero-jax")


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_x64` config option."""
    raise NotImplementedError("enable_x64 not yet implemented in zero-jax")


def explain_cache_misses(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_explain_cache_misses` config option."""
    raise NotImplementedError("explain_cache_misses not yet implemented in zero-jax")


def explicit_device_get_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_get() call."""
    raise NotImplementedError(
        "explicit_device_get_scope not yet implemented in zero-jax"
    )


def explicit_device_put_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_put*() call."""
    raise NotImplementedError(
        "explicit_device_put_scope not yet implemented in zero-jax"
    )


from . import functools


def hlo_source_file_canonicalization_regex(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_hlo_source_file_canonicalization_regex` config option."""
    raise NotImplementedError(
        "hlo_source_file_canonicalization_regex not yet implemented in zero-jax"
    )


def include_full_tracebacks_in_locations(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_include_full_tracebacks_in_locations` config option."""
    raise NotImplementedError(
        "include_full_tracebacks_in_locations not yet implemented in zero-jax"
    )


def int_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as an integer."""
    raise NotImplementedError("int_env not yet implemented in zero-jax")


from . import itertools


def jax2tf_associative_scan_reductions(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_associative_scan_reductions` config option."""
    raise NotImplementedError(
        "jax2tf_associative_scan_reductions not yet implemented in zero-jax"
    )


def jax2tf_default_native_serialization(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_default_native_serialization` config option."""
    raise NotImplementedError(
        "jax2tf_default_native_serialization not yet implemented in zero-jax"
    )


def jax_export_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_export_calling_convention_version` config option."""
    raise NotImplementedError(
        "jax_export_calling_convention_version not yet implemented in zero-jax"
    )


from . import jax_jit


def jax_pjrt_client_create_options(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pjrt_client_create_options` config option."""
    raise NotImplementedError(
        "jax_pjrt_client_create_options not yet implemented in zero-jax"
    )


def jax_platforms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_platforms` config option."""
    raise NotImplementedError("jax_platforms not yet implemented in zero-jax")


def jax_serialization_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_serialization_version` config option."""
    raise NotImplementedError(
        "jax_serialization_version not yet implemented in zero-jax"
    )


def jax_xla_profile_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_xla_profile_version` config option."""
    raise NotImplementedError("jax_xla_profile_version not yet implemented in zero-jax")


def legacy_prng_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_legacy_prng_key` config option."""
    raise NotImplementedError("legacy_prng_key not yet implemented in zero-jax")


from . import lib


def log_checkpoint_residuals(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_checkpoint_residuals` config option."""
    raise NotImplementedError(
        "log_checkpoint_residuals not yet implemented in zero-jax"
    )


def log_compiles(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_compiles` config option."""
    raise NotImplementedError("log_compiles not yet implemented in zero-jax")


logger: Any = None
from . import logging
from . import logging_config

no_default: Any = None


def numpy_dtype_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_dtype_promotion` config option."""
    raise NotImplementedError("numpy_dtype_promotion not yet implemented in zero-jax")


def numpy_rank_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_rank_promotion` config option."""
    raise NotImplementedError("numpy_rank_promotion not yet implemented in zero-jax")


from . import os


def parse_flags_with_absl(*args: Any, **kwargs: Any) -> Any:
    """Parses command-line args that start with --jax."""
    raise NotImplementedError("parse_flags_with_absl not yet implemented in zero-jax")


def persistent_cache_min_compile_time_secs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_compile_time_secs` config option."""
    raise NotImplementedError(
        "persistent_cache_min_compile_time_secs not yet implemented in zero-jax"
    )


def persistent_cache_min_entry_size_bytes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_entry_size_bytes` config option."""
    raise NotImplementedError(
        "persistent_cache_min_entry_size_bytes not yet implemented in zero-jax"
    )


def pgle_aggregation_percentile(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_aggregation_percentile` config option."""
    raise NotImplementedError(
        "pgle_aggregation_percentile not yet implemented in zero-jax"
    )


def pgle_profiling_runs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_profiling_runs` config option."""
    raise NotImplementedError("pgle_profiling_runs not yet implemented in zero-jax")


def pmap_no_rank_reduction(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_no_rank_reduction` config option."""
    raise NotImplementedError("pmap_no_rank_reduction not yet implemented in zero-jax")


def pmap_shmap_merge(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_shmap_merge` config option (transient)."""
    raise NotImplementedError("pmap_shmap_merge not yet implemented in zero-jax")


def raise_persistent_cache_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_raise_persistent_cache_errors` config option."""
    raise NotImplementedError(
        "raise_persistent_cache_errors not yet implemented in zero-jax"
    )


def random_seed_offset(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_random_seed_offset` config option."""
    raise NotImplementedError("random_seed_offset not yet implemented in zero-jax")


def remat_opt_barrier(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_remat_opt_barrier` config option."""
    raise NotImplementedError("remat_opt_barrier not yet implemented in zero-jax")


def share_autotune_config_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_autotune_config_between_hosts` config option."""
    raise NotImplementedError(
        "share_autotune_config_between_hosts not yet implemented in zero-jax"
    )


def share_binary_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts` config option."""
    raise NotImplementedError(
        "share_binary_between_hosts not yet implemented in zero-jax"
    )


def share_binary_between_hosts_timeout_ms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts_timeout_ms` config option."""
    raise NotImplementedError(
        "share_binary_between_hosts_timeout_ms not yet implemented in zero-jax"
    )


def softmax_custom_jvp(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_softmax_custom_jvp` config option (transient)."""
    raise NotImplementedError("softmax_custom_jvp not yet implemented in zero-jax")


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_spmd_mode` config option."""
    raise NotImplementedError("spmd_mode not yet implemented in zero-jax")


from . import sys
from . import threading


def threefry_gpu_kernel_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_gpu_kernel_lowering` config option."""
    raise NotImplementedError(
        "threefry_gpu_kernel_lowering not yet implemented in zero-jax"
    )


def threefry_partitionable(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_partitionable` config option (transient)."""
    raise NotImplementedError("threefry_partitionable not yet implemented in zero-jax")


def trace_context(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple of configuration values that affect tracing."""
    raise NotImplementedError("trace_context not yet implemented in zero-jax")


def traceback_filtering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_filtering` config option."""
    raise NotImplementedError("traceback_filtering not yet implemented in zero-jax")


def traceback_in_locations_limit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_in_locations_limit` config option."""
    raise NotImplementedError(
        "traceback_in_locations_limit not yet implemented in zero-jax"
    )


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    """A contextmanager to control the transfer guard level for all transfers."""
    raise NotImplementedError("transfer_guard not yet implemented in zero-jax")


def transfer_guard_device_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_device` config option."""
    raise NotImplementedError(
        "transfer_guard_device_to_device not yet implemented in zero-jax"
    )


def transfer_guard_device_to_host(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_host` config option."""
    raise NotImplementedError(
        "transfer_guard_device_to_host not yet implemented in zero-jax"
    )


def transfer_guard_host_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_host_to_device` config option."""
    raise NotImplementedError(
        "transfer_guard_host_to_device not yet implemented in zero-jax"
    )


from . import transfer_guard_lib

unset: Any = None


def update(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for update."""
    raise NotImplementedError("update not yet implemented in zero-jax")


def update_thread_local_jit_state(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for update_thread_local_jit_state."""
    raise NotImplementedError(
        "update_thread_local_jit_state not yet implemented in zero-jax"
    )


from . import xla_client


def xla_runtime_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_experimental_unsafe_xla_runtime_errors` config option."""
    raise NotImplementedError("xla_runtime_errors not yet implemented in zero-jax")
