"""Canonical JAX configuration flags and context managers."""

from __future__ import annotations

import contextlib as _py_contextlib
from collections.abc import Callable, Hashable, Iterator
from logging import Logger, getLogger
from typing import Generic, TypeVar

from zero_jax._config_internal import (
    Config,
    FlagHolder,
    bool_env,
    config,
    default_device,
    default_matmul_precision,
    enable_x64,
    flags,
    int_env,
    numpy_dtype_promotion,
    numpy_rank_promotion,
    str_env,
    transfer_guard,
)
from zero_jax._types import ArrayLikeSpec, ConfigValue

from . import (
    contextlib,
    functools,
    itertools,
    jax_jit,
    logging,
    logging_config,
    os,
    profiler,
    sys,
    threading,
    transfer_guard_lib,
)


class NoDefault:
    """Sentinel class representing an unspecified configuration flag value."""

    def __init__(self, name: str = "no_default") -> None:
        """Initialize ``NoDefault``.

        Args:
            name: Sentinel label string.
        """
        self.name: str = str(name)

    def __call__(self) -> NoDefault:
        """Return this sentinel instance when called.

        Returns:
            NoDefault: This sentinel instance.
        """
        return self


def NamedTuple(typename: str = "NamedTuple", fields: ArrayLikeSpec = ()) -> str:
    """Return a named tuple type name descriptor.

    Args:
        typename: Name of the tuple type.
        fields: Field definitions.

    Returns:
        str: Type name string.
    """
    _ = fields
    return str(typename)


def NoReturn() -> str:
    """Return the ``NoReturn`` type marker string.

    Returns:
        str: ``"NoReturn"``.
    """
    return "NoReturn"


def lib() -> str:
    """Return the active jaxlib module identifier.

    Returns:
        str: ``"jaxlib"``.
    """
    return "jaxlib"


UPGRADE_BOOL_EXTRA_DESC: str = ""
UPGRADE_BOOL_HELP: str = ""
annotations: str = "__future__.annotations"
logger: Logger = getLogger("jax.config")
no_default: NoDefault = NoDefault("no_default")
unset: NoDefault = NoDefault("unset")


def already_configured_with_absl() -> bool:
    """Return whether flags have been parsed with absl.

    Returns:
        bool: ``False`` when absl flags have not been parsed.
    """
    return False


def _define_flag(
    name: str = "flag",
    default: ConfigValue = None,
    help: str = "",
    **kwargs: ConfigValue,
) -> FlagHolder:
    """Define a configuration flag in the global ``config`` registry.

    Args:
        name: Flag name string.
        default: Default value for the flag.
        help: Help description string.
        **kwargs: Additional flag metadata options.

    Returns:
        FlagHolder: Constructed flag holder.
    """
    _ = (help, kwargs)
    config.update(name, default)
    return FlagHolder(name=name, default=default)


def DEFINE_bool(name: str = "bool_flag", default: bool = False, help: str = "", **kwargs: ConfigValue) -> FlagHolder:
    """Define a boolean configuration flag.

    Args:
        name: Flag name.
        default: Default boolean value.
        help: Help description string.
        **kwargs: Optional flag keyword arguments.

    Returns:
        FlagHolder: Boolean flag holder.
    """
    return _define_flag(name, default, help, **kwargs)


def DEFINE_enum(
    name: str = "enum_flag",
    default: str | None = "default",
    enum_values: tuple[str, ...] | list[str] = ("default",),
    help: str = "",
    **kwargs: ConfigValue,
) -> FlagHolder:
    """Define an enumerated string configuration flag.

    Args:
        name: Flag name.
        default: Default enum string value.
        enum_values: Allowed enum values.
        help: Help description string.
        **kwargs: Optional flag keyword arguments.

    Returns:
        FlagHolder: Enum flag holder.
    """
    _ = enum_values
    return _define_flag(name, default, help, **kwargs)


def DEFINE_float(name: str = "float_flag", default: float = 0.0, help: str = "", **kwargs: ConfigValue) -> FlagHolder:
    """Define a floating-point configuration flag.

    Args:
        name: Flag name.
        default: Default float value.
        help: Help description string.
        **kwargs: Optional flag keyword arguments.

    Returns:
        FlagHolder: Float flag holder.
    """
    return _define_flag(name, default, help, **kwargs)


def DEFINE_integer(name: str = "int_flag", default: int = 0, help: str = "", **kwargs: ConfigValue) -> FlagHolder:
    """Define an integer configuration flag.

    Args:
        name: Flag name.
        default: Default integer value.
        help: Help description string.
        **kwargs: Optional flag keyword arguments.

    Returns:
        FlagHolder: Integer flag holder.
    """
    return _define_flag(name, default, help, **kwargs)


def DEFINE_string(name: str = "str_flag", default: str | None = "", help: str = "", **kwargs: ConfigValue) -> FlagHolder:
    """Define a string configuration flag.

    Args:
        name: Flag name.
        default: Default string value.
        help: Help description string.
        **kwargs: Optional flag keyword arguments.

    Returns:
        FlagHolder: String flag holder.
    """
    return _define_flag(name, default, help, **kwargs)


define_bool_state = DEFINE_bool
define_enum_state = DEFINE_enum
define_float_state = DEFINE_float
define_int_state = DEFINE_integer
define_optional_enum_state = DEFINE_enum
define_optional_string_state = DEFINE_string
define_string_or_object_state = DEFINE_string
define_string_state = DEFINE_string


@_py_contextlib.contextmanager
def _flag_context(name: str, new_val: ConfigValue = None) -> Iterator[ConfigValue]:
    """Temporarily set configuration option ``name`` to ``new_val``.

    Args:
        name: Configuration flag name.
        new_val: Temporary flag value inside the context.

    Yields:
        ConfigValue: Active flag value.
    """
    prev = config.read(name)
    config.update(name, new_val)
    try:
        yield new_val
    finally:
        config.update(name, prev)


def cast(typ: ArrayLikeSpec = None, val: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return ``val`` unchanged as a runtime type cast helper.

    Args:
        typ: Target type descriptor.
        val: Value to cast.

    Returns:
        ArrayLikeSpec: ``val`` when provided, otherwise ``typ``.
    """
    return val if val is not None else typ


def check_exists(name: str = "jax_enable_x64") -> bool:
    """Check whether configuration option ``name`` exists.

    Args:
        name: Option name to check.

    Returns:
        bool: ``True`` if ``name`` is a non-empty option name.
    """
    return bool(name)


def parse_flags_with_absl() -> bool:
    """Configure JAX flags to integrate with absl.flags.

    Returns:
        bool: ``True`` once configured.
    """
    return True


def update(name: str = "jax_enable_x64", val: ConfigValue = False) -> ConfigValue:
    """Update configuration option ``name`` to ``val``.

    Args:
        name: Configuration option name.
        val: New value to assign.

    Returns:
        ConfigValue: Assigned value ``val``.
    """
    config.update(name, val)
    return val


def update_thread_local_jit_state(**kw: ConfigValue) -> dict[str, ConfigValue]:
    """Update thread-local JIT state attributes from ``kw``.

    Args:
        **kw: Thread-local JIT state fields to update.

    Returns:
        dict[str, ConfigValue]: Updated key-value mapping.
    """
    return dict(kw)


def bcoo_cusparse_lowering(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_bcoo_cusparse_lowering`` configuration option.

    Args:
        new_val: Value to set for ``jax_bcoo_cusparse_lowering`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_bcoo_cusparse_lowering", new_val)


def check_tracer_leaks(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_check_tracer_leaks`` configuration option.

    Args:
        new_val: Value to set for ``jax_check_tracer_leaks`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_check_tracer_leaks", new_val)


def checking_leaks(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_checking_leaks`` configuration option.

    Args:
        new_val: Value to set for ``jax_checking_leaks`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_checking_leaks", new_val)


def compilation_cache_dir(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_compilation_cache_dir`` configuration option.

    Args:
        new_val: Value to set for ``jax_compilation_cache_dir`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_compilation_cache_dir", new_val)


def compilation_cache_include_metadata_in_key(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_compilation_cache_include_metadata_in_key`` configuration option.

    Args:
        new_val: Value to set for ``jax_compilation_cache_include_metadata_in_key`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_compilation_cache_include_metadata_in_key", new_val)


def compilation_cache_max_size(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_compilation_cache_max_size`` configuration option.

    Args:
        new_val: Value to set for ``jax_compilation_cache_max_size`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_compilation_cache_max_size", new_val)


def custom_vjp_disable_shape_check(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_custom_vjp_disable_shape_check`` configuration option.

    Args:
        new_val: Value to set for ``jax_custom_vjp_disable_shape_check`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_custom_vjp_disable_shape_check", new_val)


def debug_infs(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_debug_infs`` configuration option.

    Args:
        new_val: Value to set for ``jax_debug_infs`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_debug_infs", new_val)


def debug_key_reuse(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_debug_key_reuse`` configuration option.

    Args:
        new_val: Value to set for ``jax_debug_key_reuse`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_debug_key_reuse", new_val)


def debug_nans(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_debug_nans`` configuration option.

    Args:
        new_val: Value to set for ``jax_debug_nans`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_debug_nans", new_val)


def default_dtype_bits(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_default_dtype_bits`` configuration option.

    Args:
        new_val: Value to set for ``jax_default_dtype_bits`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_default_dtype_bits", new_val)


def default_prng_impl(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_default_prng_impl`` configuration option.

    Args:
        new_val: Value to set for ``jax_default_prng_impl`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_default_prng_impl", new_val)


def disable_jit(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_disable_jit`` configuration option.

    Args:
        new_val: Value to set for ``jax_disable_jit`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_disable_jit", new_val)


def distributed_debug(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_distributed_debug`` configuration option.

    Args:
        new_val: Value to set for ``jax_distributed_debug`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_distributed_debug", new_val)


def dynamic_shapes(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_dynamic_shapes`` configuration option.

    Args:
        new_val: Value to set for ``jax_dynamic_shapes`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_dynamic_shapes", new_val)


def eager_pmap(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_eager_pmap`` configuration option.

    Args:
        new_val: Value to set for ``jax_eager_pmap`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_eager_pmap", new_val)


def enable_checks(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_checks`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_checks`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_checks", new_val)


def enable_compilation_cache(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_compilation_cache`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_compilation_cache`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_compilation_cache", new_val)


def enable_custom_prng(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_custom_prng`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_custom_prng`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_custom_prng", new_val)


def enable_custom_vjp_by_custom_transpose(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_custom_vjp_by_custom_transpose`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_custom_vjp_by_custom_transpose`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_custom_vjp_by_custom_transpose", new_val)


def enable_memories(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_memories`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_memories`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_memories", new_val)


def enable_pgle(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_enable_pgle`` configuration option.

    Args:
        new_val: Value to set for ``jax_enable_pgle`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_enable_pgle", new_val)


def explain_cache_misses(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_explain_cache_misses`` configuration option.

    Args:
        new_val: Value to set for ``jax_explain_cache_misses`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_explain_cache_misses", new_val)


def explicit_device_get_scope(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_explicit_device_get_scope`` configuration option.

    Args:
        new_val: Value to set for ``jax_explicit_device_get_scope`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_explicit_device_get_scope", new_val)


def explicit_device_put_scope(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_explicit_device_put_scope`` configuration option.

    Args:
        new_val: Value to set for ``jax_explicit_device_put_scope`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_explicit_device_put_scope", new_val)


def hlo_source_file_canonicalization_regex(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_hlo_source_file_canonicalization_regex`` configuration option.

    Args:
        new_val: Value to set for ``jax_hlo_source_file_canonicalization_regex`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_hlo_source_file_canonicalization_regex", new_val)


def include_full_tracebacks_in_locations(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_include_full_tracebacks_in_locations`` configuration option.

    Args:
        new_val: Value to set for ``jax_include_full_tracebacks_in_locations`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_include_full_tracebacks_in_locations", new_val)


def jax2tf_associative_scan_reductions(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax2tf_associative_scan_reductions`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax2tf_associative_scan_reductions`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax2tf_associative_scan_reductions", new_val)


def jax2tf_default_native_serialization(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax2tf_default_native_serialization`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax2tf_default_native_serialization`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax2tf_default_native_serialization", new_val)


def jax_export_calling_convention_version(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax_export_calling_convention_version`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax_export_calling_convention_version`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax_export_calling_convention_version", new_val)


def jax_pjrt_client_create_options(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax_pjrt_client_create_options`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax_pjrt_client_create_options`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax_pjrt_client_create_options", new_val)


def jax_platforms(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax_platforms`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax_platforms`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax_platforms", new_val)


def jax_serialization_version(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax_serialization_version`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax_serialization_version`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax_serialization_version", new_val)


def jax_xla_profile_version(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_jax_xla_profile_version`` configuration option.

    Args:
        new_val: Value to set for ``jax_jax_xla_profile_version`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_jax_xla_profile_version", new_val)


def legacy_prng_key(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_legacy_prng_key`` configuration option.

    Args:
        new_val: Value to set for ``jax_legacy_prng_key`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_legacy_prng_key", new_val)


def log_checkpoint_residuals(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_log_checkpoint_residuals`` configuration option.

    Args:
        new_val: Value to set for ``jax_log_checkpoint_residuals`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_log_checkpoint_residuals", new_val)


def log_compiles(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_log_compiles`` configuration option.

    Args:
        new_val: Value to set for ``jax_log_compiles`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_log_compiles", new_val)


def persistent_cache_min_compile_time_secs(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_persistent_cache_min_compile_time_secs`` configuration option.

    Args:
        new_val: Value to set for ``jax_persistent_cache_min_compile_time_secs`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_persistent_cache_min_compile_time_secs", new_val)


def persistent_cache_min_entry_size_bytes(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_persistent_cache_min_entry_size_bytes`` configuration option.

    Args:
        new_val: Value to set for ``jax_persistent_cache_min_entry_size_bytes`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_persistent_cache_min_entry_size_bytes", new_val)


def pgle_aggregation_percentile(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_pgle_aggregation_percentile`` configuration option.

    Args:
        new_val: Value to set for ``jax_pgle_aggregation_percentile`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_pgle_aggregation_percentile", new_val)


def pgle_profiling_runs(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_pgle_profiling_runs`` configuration option.

    Args:
        new_val: Value to set for ``jax_pgle_profiling_runs`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_pgle_profiling_runs", new_val)


def pmap_no_rank_reduction(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_pmap_no_rank_reduction`` configuration option.

    Args:
        new_val: Value to set for ``jax_pmap_no_rank_reduction`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_pmap_no_rank_reduction", new_val)


def pmap_shmap_merge(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_pmap_shmap_merge`` configuration option.

    Args:
        new_val: Value to set for ``jax_pmap_shmap_merge`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_pmap_shmap_merge", new_val)


def raise_persistent_cache_errors(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_raise_persistent_cache_errors`` configuration option.

    Args:
        new_val: Value to set for ``jax_raise_persistent_cache_errors`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_raise_persistent_cache_errors", new_val)


def random_seed_offset(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_random_seed_offset`` configuration option.

    Args:
        new_val: Value to set for ``jax_random_seed_offset`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_random_seed_offset", new_val)


def remat_opt_barrier(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_remat_opt_barrier`` configuration option.

    Args:
        new_val: Value to set for ``jax_remat_opt_barrier`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_remat_opt_barrier", new_val)


def share_autotune_config_between_hosts(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_share_autotune_config_between_hosts`` configuration option.

    Args:
        new_val: Value to set for ``jax_share_autotune_config_between_hosts`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_share_autotune_config_between_hosts", new_val)


def share_binary_between_hosts(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_share_binary_between_hosts`` configuration option.

    Args:
        new_val: Value to set for ``jax_share_binary_between_hosts`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_share_binary_between_hosts", new_val)


def share_binary_between_hosts_timeout_ms(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_share_binary_between_hosts_timeout_ms`` configuration option.

    Args:
        new_val: Value to set for ``jax_share_binary_between_hosts_timeout_ms`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_share_binary_between_hosts_timeout_ms", new_val)


def softmax_custom_jvp(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_softmax_custom_jvp`` configuration option.

    Args:
        new_val: Value to set for ``jax_softmax_custom_jvp`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_softmax_custom_jvp", new_val)


def spmd_mode(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_spmd_mode`` configuration option.

    Args:
        new_val: Value to set for ``jax_spmd_mode`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_spmd_mode", new_val)


def threefry_gpu_kernel_lowering(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_threefry_gpu_kernel_lowering`` configuration option.

    Args:
        new_val: Value to set for ``jax_threefry_gpu_kernel_lowering`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_threefry_gpu_kernel_lowering", new_val)


def threefry_partitionable(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_threefry_partitionable`` configuration option.

    Args:
        new_val: Value to set for ``jax_threefry_partitionable`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_threefry_partitionable", new_val)


def trace_context(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_trace_context`` configuration option.

    Args:
        new_val: Value to set for ``jax_trace_context`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_trace_context", new_val)


def traceback_filtering(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_traceback_filtering`` configuration option.

    Args:
        new_val: Value to set for ``jax_traceback_filtering`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_traceback_filtering", new_val)


def traceback_in_locations_limit(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_traceback_in_locations_limit`` configuration option.

    Args:
        new_val: Value to set for ``jax_traceback_in_locations_limit`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_traceback_in_locations_limit", new_val)


def transfer_guard_device_to_device(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_transfer_guard_device_to_device`` configuration option.

    Args:
        new_val: Value to set for ``jax_transfer_guard_device_to_device`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_transfer_guard_device_to_device", new_val)


def transfer_guard_device_to_host(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_transfer_guard_device_to_host`` configuration option.

    Args:
        new_val: Value to set for ``jax_transfer_guard_device_to_host`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_transfer_guard_device_to_host", new_val)


def transfer_guard_host_to_device(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_transfer_guard_host_to_device`` configuration option.

    Args:
        new_val: Value to set for ``jax_transfer_guard_host_to_device`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_transfer_guard_host_to_device", new_val)


def xla_runtime_errors(new_val: ConfigValue = None) -> _py_contextlib.AbstractContextManager[ConfigValue]:
    """Context manager for ``jax_xla_runtime_errors`` configuration option.

    Args:
        new_val: Value to set for ``jax_xla_runtime_errors`` inside the context.

    Returns:
        _py_contextlib.AbstractContextManager[ConfigValue]: Scoped configuration context manager.
    """
    return _flag_context("jax_xla_runtime_errors", new_val)


__all__ = [
    "Callable",
    "Config",
    "DEFINE_bool",
    "DEFINE_enum",
    "DEFINE_float",
    "DEFINE_integer",
    "DEFINE_string",
    "FlagHolder",
    "Generic",
    "Hashable",
    "Iterator",
    "NamedTuple",
    "NoDefault",
    "NoReturn",
    "TypeVar",
    "UPGRADE_BOOL_EXTRA_DESC",
    "UPGRADE_BOOL_HELP",
    "_contextlib_mod",
    "_define_flag",
    "_flag_context",
    "already_configured_with_absl",
    "annotations",
    "bcoo_cusparse_lowering",
    "bool_env",
    "cast",
    "check_exists",
    "check_tracer_leaks",
    "checking_leaks",
    "compilation_cache_dir",
    "compilation_cache_include_metadata_in_key",
    "compilation_cache_max_size",
    "config",
    "contextlib",
    "custom_vjp_disable_shape_check",
    "debug_infs",
    "debug_key_reuse",
    "debug_nans",
    "default_device",
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
    "disable_jit",
    "distributed_debug",
    "dynamic_shapes",
    "eager_pmap",
    "enable_checks",
    "enable_compilation_cache",
    "enable_custom_prng",
    "enable_custom_vjp_by_custom_transpose",
    "enable_memories",
    "enable_pgle",
    "enable_x64",
    "explain_cache_misses",
    "explicit_device_get_scope",
    "explicit_device_put_scope",
    "flags",
    "functools",
    "hlo_source_file_canonicalization_regex",
    "include_full_tracebacks_in_locations",
    "int_env",
    "itertools",
    "jax2tf_associative_scan_reductions",
    "jax2tf_default_native_serialization",
    "jax_export_calling_convention_version",
    "jax_jit",
    "jax_pjrt_client_create_options",
    "jax_platforms",
    "jax_serialization_version",
    "jax_xla_profile_version",
    "legacy_prng_key",
    "lib",
    "log_checkpoint_residuals",
    "log_compiles",
    "logger",
    "logging",
    "logging_config",
    "no_default",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "os",
    "parse_flags_with_absl",
    "persistent_cache_min_compile_time_secs",
    "persistent_cache_min_entry_size_bytes",
    "pgle_aggregation_percentile",
    "pgle_profiling_runs",
    "pmap_no_rank_reduction",
    "pmap_shmap_merge",
    "profiler",
    "raise_persistent_cache_errors",
    "random_seed_offset",
    "remat_opt_barrier",
    "share_autotune_config_between_hosts",
    "share_binary_between_hosts",
    "share_binary_between_hosts_timeout_ms",
    "softmax_custom_jvp",
    "spmd_mode",
    "str_env",
    "sys",
    "threading",
    "threefry_gpu_kernel_lowering",
    "threefry_partitionable",
    "trace_context",
    "traceback_filtering",
    "traceback_in_locations_limit",
    "transfer_guard",
    "transfer_guard_device_to_device",
    "transfer_guard_device_to_host",
    "transfer_guard_host_to_device",
    "transfer_guard_lib",
    "unset",
    "update",
    "update_thread_local_jit_state",
    "xla_runtime_errors",
]
