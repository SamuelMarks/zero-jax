"""Frontend API routing for jax.experimental.x64_context.config."""

from typing import Any
from typing import Callable
import ml_switcheroo_compiler.ops as _ops


class Config:
    """Mock implementation for Config."""

    pass


def DEFINE_bool(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_bool."""
    return getattr(_ops, "DEFINE_bool")(*args, **kwargs)


def DEFINE_enum(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_enum."""
    return getattr(_ops, "DEFINE_enum")(*args, **kwargs)


def DEFINE_float(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_float."""
    return getattr(_ops, "DEFINE_float")(*args, **kwargs)


def DEFINE_integer(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_integer."""
    return getattr(_ops, "DEFINE_integer")(*args, **kwargs)


def DEFINE_string(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DEFINE_string."""
    return getattr(_ops, "DEFINE_string")(*args, **kwargs)


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
    return getattr(_ops, "NamedTuple")(*args, **kwargs)


class NoDefault:
    """Mock implementation for NoDefault."""

    pass


def NoReturn(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating functions that never return."""
    return getattr(_ops, "NoReturn")(*args, **kwargs)


class TypeVar:
    """Type variable."""

    pass


UPGRADE_BOOL_EXTRA_DESC: Any = None

UPGRADE_BOOL_HELP: Any = None

already_configured_with_absl: Any = None

annotations: Any = None


def bcoo_cusparse_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_bcoo_cusparse_lowering` config option."""
    return getattr(_ops, "bcoo_cusparse_lowering")(*args, **kwargs)


def bool_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as a boolean."""
    return getattr(_ops, "bool_env")(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return getattr(_ops, "cast")(*args, **kwargs)


def check_exists(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_exists."""
    return getattr(_ops, "check_exists")(*args, **kwargs)


def check_tracer_leaks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_check_tracer_leaks` config option."""
    return getattr(_ops, "check_tracer_leaks")(*args, **kwargs)


def checking_leaks(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "checking_leaks")(*args, **kwargs)


def compilation_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_dir` config option."""
    return getattr(_ops, "compilation_cache_dir")(*args, **kwargs)


def compilation_cache_include_metadata_in_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_include_metadata_in_key` config option."""
    return getattr(_ops, "compilation_cache_include_metadata_in_key")(*args, **kwargs)


def compilation_cache_max_size(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_compilation_cache_max_size` config option."""
    return getattr(_ops, "compilation_cache_max_size")(*args, **kwargs)


config: Any = None
from . import contextlib


def custom_vjp_disable_shape_check(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_custom_vjp_disable_shape_check` config option (transient)."""
    return getattr(_ops, "custom_vjp_disable_shape_check")(*args, **kwargs)


def debug_infs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_infs` config option."""
    return getattr(_ops, "debug_infs")(*args, **kwargs)


def debug_key_reuse(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_key_reuse` config option."""
    return getattr(_ops, "debug_key_reuse")(*args, **kwargs)


def debug_nans(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_debug_nans` config option."""
    return getattr(_ops, "debug_nans")(*args, **kwargs)


def default_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_device` config option."""
    return getattr(_ops, "default_device")(*args, **kwargs)


def default_dtype_bits(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_dtype_bits` config option."""
    return getattr(_ops, "default_dtype_bits")(*args, **kwargs)


def default_matmul_precision(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_matmul_precision` config option."""
    return getattr(_ops, "default_matmul_precision")(*args, **kwargs)


def default_prng_impl(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_default_prng_impl` config option."""
    return getattr(_ops, "default_prng_impl")(*args, **kwargs)


def define_bool_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_bool_state")(*args, **kwargs)


def define_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_enum_state")(*args, **kwargs)


def define_float_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_float_state")(*args, **kwargs)


def define_int_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_int_state")(*args, **kwargs)


def define_optional_enum_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_optional_enum_state")(*args, **kwargs)


def define_optional_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_optional_string_state")(*args, **kwargs)


def define_string_or_object_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_string_or_object_state")(*args, **kwargs)


def define_string_state(*args: Any, **kwargs: Any) -> Any:
    """Set up thread-local state and return a contextmanager for managing it."""
    return getattr(_ops, "define_string_state")(*args, **kwargs)


def disable_jit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_disable_jit` config option."""
    return getattr(_ops, "disable_jit")(*args, **kwargs)


def distributed_debug(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_distributed_debug` config option."""
    return getattr(_ops, "distributed_debug")(*args, **kwargs)


def dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_dynamic_shapes` config option."""
    return getattr(_ops, "dynamic_shapes")(*args, **kwargs)


def eager_pmap(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_eager_pmap` config option (transient)."""
    return getattr(_ops, "eager_pmap")(*args, **kwargs)


def enable_checks(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_checks` config option."""
    return getattr(_ops, "enable_checks")(*args, **kwargs)


def enable_compilation_cache(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_compilation_cache` config option."""
    return getattr(_ops, "enable_compilation_cache")(*args, **kwargs)


def enable_custom_prng(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_prng` config option (transient)."""
    return getattr(_ops, "enable_custom_prng")(*args, **kwargs)


def enable_custom_vjp_by_custom_transpose(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_custom_vjp_by_custom_transpose` config option (transient)."""
    return getattr(_ops, "enable_custom_vjp_by_custom_transpose")(*args, **kwargs)


def enable_memories(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_memories` config option (transient)."""
    return getattr(_ops, "enable_memories")(*args, **kwargs)


def enable_pgle(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_pgle` config option."""
    return getattr(_ops, "enable_pgle")(*args, **kwargs)


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_enable_x64` config option."""
    return getattr(_ops, "enable_x64")(*args, **kwargs)


def explain_cache_misses(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_explain_cache_misses` config option."""
    return getattr(_ops, "explain_cache_misses")(*args, **kwargs)


def explicit_device_get_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_get() call."""
    return getattr(_ops, "explicit_device_get_scope")(*args, **kwargs)


def explicit_device_put_scope(*args: Any, **kwargs: Any) -> Any:
    """Indicates that the current context is an explicit device_put*() call."""
    return getattr(_ops, "explicit_device_put_scope")(*args, **kwargs)


from . import functools


def hlo_source_file_canonicalization_regex(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_hlo_source_file_canonicalization_regex` config option."""
    return getattr(_ops, "hlo_source_file_canonicalization_regex")(*args, **kwargs)


def include_full_tracebacks_in_locations(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_include_full_tracebacks_in_locations` config option."""
    return getattr(_ops, "include_full_tracebacks_in_locations")(*args, **kwargs)


def int_env(*args: Any, **kwargs: Any) -> Any:
    """Read an environment variable and interpret it as an integer."""
    return getattr(_ops, "int_env")(*args, **kwargs)


from . import itertools


def jax2tf_associative_scan_reductions(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_associative_scan_reductions` config option."""
    return getattr(_ops, "jax2tf_associative_scan_reductions")(*args, **kwargs)


def jax2tf_default_native_serialization(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax2tf_default_native_serialization` config option."""
    return getattr(_ops, "jax2tf_default_native_serialization")(*args, **kwargs)


def jax_export_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_export_calling_convention_version` config option."""
    return getattr(_ops, "jax_export_calling_convention_version")(*args, **kwargs)


from . import jax_jit


def jax_pjrt_client_create_options(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pjrt_client_create_options` config option."""
    return getattr(_ops, "jax_pjrt_client_create_options")(*args, **kwargs)


def jax_platforms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_platforms` config option."""
    return getattr(_ops, "jax_platforms")(*args, **kwargs)


def jax_serialization_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_serialization_version` config option."""
    return getattr(_ops, "jax_serialization_version")(*args, **kwargs)


def jax_xla_profile_version(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_xla_profile_version` config option."""
    return getattr(_ops, "jax_xla_profile_version")(*args, **kwargs)


def legacy_prng_key(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_legacy_prng_key` config option."""
    return getattr(_ops, "legacy_prng_key")(*args, **kwargs)


def log_checkpoint_residuals(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_checkpoint_residuals` config option."""
    return getattr(_ops, "log_checkpoint_residuals")(*args, **kwargs)


def log_compiles(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_log_compiles` config option."""
    return getattr(_ops, "log_compiles")(*args, **kwargs)


logger: Any = None
from . import logging
from . import logging_config

no_default: Any = None


def numpy_dtype_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_dtype_promotion` config option."""
    return getattr(_ops, "numpy_dtype_promotion")(*args, **kwargs)


def numpy_rank_promotion(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_numpy_rank_promotion` config option."""
    return getattr(_ops, "numpy_rank_promotion")(*args, **kwargs)


from . import os


def parse_flags_with_absl(*args: Any, **kwargs: Any) -> Any:
    """Parses command-line args that start with --jax."""
    return getattr(_ops, "parse_flags_with_absl")(*args, **kwargs)


def persistent_cache_min_compile_time_secs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_compile_time_secs` config option."""
    return getattr(_ops, "persistent_cache_min_compile_time_secs")(*args, **kwargs)


def persistent_cache_min_entry_size_bytes(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_persistent_cache_min_entry_size_bytes` config option."""
    return getattr(_ops, "persistent_cache_min_entry_size_bytes")(*args, **kwargs)


def pgle_aggregation_percentile(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_aggregation_percentile` config option."""
    return getattr(_ops, "pgle_aggregation_percentile")(*args, **kwargs)


def pgle_profiling_runs(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pgle_profiling_runs` config option."""
    return getattr(_ops, "pgle_profiling_runs")(*args, **kwargs)


def pmap_no_rank_reduction(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_no_rank_reduction` config option."""
    return getattr(_ops, "pmap_no_rank_reduction")(*args, **kwargs)


def pmap_shmap_merge(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_pmap_shmap_merge` config option (transient)."""
    return getattr(_ops, "pmap_shmap_merge")(*args, **kwargs)


def raise_persistent_cache_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_raise_persistent_cache_errors` config option."""
    return getattr(_ops, "raise_persistent_cache_errors")(*args, **kwargs)


def random_seed_offset(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_random_seed_offset` config option."""
    return getattr(_ops, "random_seed_offset")(*args, **kwargs)


def remat_opt_barrier(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_remat_opt_barrier` config option."""
    return getattr(_ops, "remat_opt_barrier")(*args, **kwargs)


def share_autotune_config_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_autotune_config_between_hosts` config option."""
    return getattr(_ops, "share_autotune_config_between_hosts")(*args, **kwargs)


def share_binary_between_hosts(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts` config option."""
    return getattr(_ops, "share_binary_between_hosts")(*args, **kwargs)


def share_binary_between_hosts_timeout_ms(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_share_binary_between_hosts_timeout_ms` config option."""
    return getattr(_ops, "share_binary_between_hosts_timeout_ms")(*args, **kwargs)


def softmax_custom_jvp(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_softmax_custom_jvp` config option (transient)."""
    return getattr(_ops, "softmax_custom_jvp")(*args, **kwargs)


def spmd_mode(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_spmd_mode` config option."""
    return getattr(_ops, "spmd_mode")(*args, **kwargs)


from . import sys
from . import threading


def threefry_gpu_kernel_lowering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_gpu_kernel_lowering` config option."""
    return getattr(_ops, "threefry_gpu_kernel_lowering")(*args, **kwargs)


def threefry_partitionable(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_threefry_partitionable` config option (transient)."""
    return getattr(_ops, "threefry_partitionable")(*args, **kwargs)


def trace_context(*args: Any, **kwargs: Any) -> Any:
    """Returns a tuple of configuration values that affect tracing."""
    return getattr(_ops, "trace_context")(*args, **kwargs)


def traceback_filtering(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_filtering` config option."""
    return getattr(_ops, "traceback_filtering")(*args, **kwargs)


def traceback_in_locations_limit(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_traceback_in_locations_limit` config option."""
    return getattr(_ops, "traceback_in_locations_limit")(*args, **kwargs)


def transfer_guard(*args: Any, **kwargs: Any) -> Any:
    """A contextmanager to control the transfer guard level for all transfers."""
    return getattr(_ops, "transfer_guard")(*args, **kwargs)


def transfer_guard_device_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_device` config option."""
    return getattr(_ops, "transfer_guard_device_to_device")(*args, **kwargs)


def transfer_guard_device_to_host(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_device_to_host` config option."""
    return getattr(_ops, "transfer_guard_device_to_host")(*args, **kwargs)


def transfer_guard_host_to_device(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_transfer_guard_host_to_device` config option."""
    return getattr(_ops, "transfer_guard_host_to_device")(*args, **kwargs)


from . import transfer_guard_lib

unset: Any = None


def update(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for update."""
    return getattr(_ops, "update")(*args, **kwargs)


def update_thread_local_jit_state(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for update_thread_local_jit_state."""
    return getattr(_ops, "update_thread_local_jit_state")(*args, **kwargs)


from . import xla_client


def xla_runtime_errors(*args: Any, **kwargs: Any) -> Any:
    """Context manager for `jax_experimental_unsafe_xla_runtime_errors` config option."""
    return getattr(_ops, "xla_runtime_errors")(*args, **kwargs)
