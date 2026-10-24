"""Frontend API routing for jax.experimental.x64_context.config."""

from __future__ import annotations

from collections.abc import Callable

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class Config:
    """Frontend state holder for Config."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def DEFINE_bool(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for DEFINE_bool.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.DEFINE_bool(*args, **kwargs)


def DEFINE_enum(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for DEFINE_enum.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.DEFINE_enum(*args, **kwargs)


def DEFINE_float(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for DEFINE_float.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.DEFINE_float(*args, **kwargs)


def DEFINE_integer(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for DEFINE_integer.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.DEFINE_integer(*args, **kwargs)


def DEFINE_string(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for DEFINE_string.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.DEFINE_string(*args, **kwargs)


class FlagHolder:
    """Frontend state holder for FlagHolder."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


class Generic:
    """Abstract base class for generic types."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


class Hashable:
    """Frontend state holder for Hashable."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


class Iterator:
    """Frontend state holder for Iterator."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def NamedTuple(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Typed version of namedtuple.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.NamedTuple(*args, **kwargs)


class NoDefault:
    """Frontend state holder for NoDefault."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def NoReturn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating functions that never return.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.NoReturn(*args, **kwargs)


class TypeVar:
    """Type variable."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


UPGRADE_BOOL_EXTRA_DESC = ""

UPGRADE_BOOL_HELP = ""


def already_configured_with_absl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``already_configured_with_absl``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


annotations: ArrayLikeSpec = None


def bcoo_cusparse_lowering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_bcoo_cusparse_lowering` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.bcoo_cusparse_lowering(*args, **kwargs)


def bool_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Read an environment variable and interpret it as a boolean.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.bool_env(*args, **kwargs)


def cast(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cast a value to a type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cast(*args, **kwargs)


def check_exists(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for check_exists.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.check_exists(*args, **kwargs)


def check_tracer_leaks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_check_tracer_leaks` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.check_tracer_leaks(*args, **kwargs)


def checking_leaks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.checking_leaks(*args, **kwargs)


def compilation_cache_dir(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_compilation_cache_dir` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.compilation_cache_dir(*args, **kwargs)


def compilation_cache_include_metadata_in_key(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_compilation_cache_include_metadata_in_key` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.compilation_cache_include_metadata_in_key(*args, **kwargs)


def compilation_cache_max_size(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_compilation_cache_max_size` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.compilation_cache_max_size(*args, **kwargs)


def config(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``config``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


from . import contextlib


def custom_vjp_disable_shape_check(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_custom_vjp_disable_shape_check` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.custom_vjp_disable_shape_check(*args, **kwargs)


def debug_infs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_debug_infs` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.debug_infs(*args, **kwargs)


def debug_key_reuse(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_debug_key_reuse` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.debug_key_reuse(*args, **kwargs)


def debug_nans(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_debug_nans` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.debug_nans(*args, **kwargs)


def default_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_default_device` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.default_device(*args, **kwargs)


def default_dtype_bits(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_default_dtype_bits` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.default_dtype_bits(*args, **kwargs)


def default_matmul_precision(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_default_matmul_precision` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.default_matmul_precision(*args, **kwargs)


def default_prng_impl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_default_prng_impl` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.default_prng_impl(*args, **kwargs)


def define_bool_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_bool_state(*args, **kwargs)


def define_enum_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_enum_state(*args, **kwargs)


def define_float_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_float_state(*args, **kwargs)


def define_int_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_int_state(*args, **kwargs)


def define_optional_enum_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_optional_enum_state(*args, **kwargs)


def define_optional_string_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_optional_string_state(*args, **kwargs)


def define_string_or_object_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_string_or_object_state(*args, **kwargs)


def define_string_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Set up thread-local state and return a contextmanager for managing it.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.define_string_state(*args, **kwargs)


def disable_jit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_disable_jit` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.disable_jit(*args, **kwargs)


def distributed_debug(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_distributed_debug` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.distributed_debug(*args, **kwargs)


def dynamic_shapes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_dynamic_shapes` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.dynamic_shapes(*args, **kwargs)


def eager_pmap(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_eager_pmap` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.eager_pmap(*args, **kwargs)


def enable_checks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_checks` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_checks(*args, **kwargs)


def enable_compilation_cache(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_compilation_cache` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_compilation_cache(*args, **kwargs)


def enable_custom_prng(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_custom_prng` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_custom_prng(*args, **kwargs)


def enable_custom_vjp_by_custom_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_custom_vjp_by_custom_transpose` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_custom_vjp_by_custom_transpose(*args, **kwargs)


def enable_memories(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_memories` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_memories(*args, **kwargs)


def enable_pgle(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_pgle` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_pgle(*args, **kwargs)


def enable_x64(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_enable_x64` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_x64(*args, **kwargs)


def explain_cache_misses(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_explain_cache_misses` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.explain_cache_misses(*args, **kwargs)


def explicit_device_get_scope(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Indicates that the current context is an explicit device_get() call.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.explicit_device_get_scope(*args, **kwargs)


def explicit_device_put_scope(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Indicates that the current context is an explicit device_put*() call.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.explicit_device_put_scope(*args, **kwargs)


from . import functools


def hlo_source_file_canonicalization_regex(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_hlo_source_file_canonicalization_regex` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.hlo_source_file_canonicalization_regex(*args, **kwargs)


def include_full_tracebacks_in_locations(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_include_full_tracebacks_in_locations` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.include_full_tracebacks_in_locations(*args, **kwargs)


def int_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Read an environment variable and interpret it as an integer.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.int_env(*args, **kwargs)


from . import itertools


def jax2tf_associative_scan_reductions(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax2tf_associative_scan_reductions` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax2tf_associative_scan_reductions(*args, **kwargs)


def jax2tf_default_native_serialization(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax2tf_default_native_serialization` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax2tf_default_native_serialization(*args, **kwargs)


def jax_export_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_export_calling_convention_version` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax_export_calling_convention_version(*args, **kwargs)


from . import jax_jit


def jax_pjrt_client_create_options(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_pjrt_client_create_options` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax_pjrt_client_create_options(*args, **kwargs)


def jax_platforms(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_platforms` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax_platforms(*args, **kwargs)


def jax_serialization_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_serialization_version` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax_serialization_version(*args, **kwargs)


def jax_xla_profile_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_xla_profile_version` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jax_xla_profile_version(*args, **kwargs)


def legacy_prng_key(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_legacy_prng_key` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.legacy_prng_key(*args, **kwargs)


def log_checkpoint_residuals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_log_checkpoint_residuals` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.log_checkpoint_residuals(*args, **kwargs)


def log_compiles(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_log_compiles` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.log_compiles(*args, **kwargs)


def logger(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``logger``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


from . import logging, logging_config


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
    """Context manager for `jax_numpy_dtype_promotion` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.numpy_dtype_promotion(*args, **kwargs)


def numpy_rank_promotion(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_numpy_rank_promotion` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.numpy_rank_promotion(*args, **kwargs)


from . import os


def parse_flags_with_absl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Parses command-line args that start with --jax.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.parse_flags_with_absl(*args, **kwargs)


def persistent_cache_min_compile_time_secs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_persistent_cache_min_compile_time_secs` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.persistent_cache_min_compile_time_secs(*args, **kwargs)


def persistent_cache_min_entry_size_bytes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_persistent_cache_min_entry_size_bytes` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.persistent_cache_min_entry_size_bytes(*args, **kwargs)


def pgle_aggregation_percentile(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_pgle_aggregation_percentile` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.pgle_aggregation_percentile(*args, **kwargs)


def pgle_profiling_runs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_pgle_profiling_runs` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.pgle_profiling_runs(*args, **kwargs)


def pmap_no_rank_reduction(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_pmap_no_rank_reduction` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.pmap_no_rank_reduction(*args, **kwargs)


def pmap_shmap_merge(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_pmap_shmap_merge` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.pmap_shmap_merge(*args, **kwargs)


def raise_persistent_cache_errors(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_raise_persistent_cache_errors` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.raise_persistent_cache_errors(*args, **kwargs)


def random_seed_offset(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_random_seed_offset` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.random_seed_offset(*args, **kwargs)


def remat_opt_barrier(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_remat_opt_barrier` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.remat_opt_barrier(*args, **kwargs)


def share_autotune_config_between_hosts(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_share_autotune_config_between_hosts` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.share_autotune_config_between_hosts(*args, **kwargs)


def share_binary_between_hosts(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_share_binary_between_hosts` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.share_binary_between_hosts(*args, **kwargs)


def share_binary_between_hosts_timeout_ms(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_share_binary_between_hosts_timeout_ms` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.share_binary_between_hosts_timeout_ms(*args, **kwargs)


def softmax_custom_jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_softmax_custom_jvp` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.softmax_custom_jvp(*args, **kwargs)


def spmd_mode(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_spmd_mode` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.spmd_mode(*args, **kwargs)


from . import sys, threading


def threefry_gpu_kernel_lowering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_threefry_gpu_kernel_lowering` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.threefry_gpu_kernel_lowering(*args, **kwargs)


def threefry_partitionable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_threefry_partitionable` config option (transient).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.threefry_partitionable(*args, **kwargs)


def trace_context(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a tuple of configuration values that affect tracing.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.trace_context(*args, **kwargs)


def traceback_filtering(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_traceback_filtering` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.traceback_filtering(*args, **kwargs)


def traceback_in_locations_limit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_traceback_in_locations_limit` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.traceback_in_locations_limit(*args, **kwargs)


def transfer_guard(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """A contextmanager to control the transfer guard level for all transfers.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.transfer_guard(*args, **kwargs)


def transfer_guard_device_to_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_transfer_guard_device_to_device` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.transfer_guard_device_to_device(*args, **kwargs)


def transfer_guard_device_to_host(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_transfer_guard_device_to_host` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.transfer_guard_device_to_host(*args, **kwargs)


def transfer_guard_host_to_device(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_transfer_guard_host_to_device` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.transfer_guard_host_to_device(*args, **kwargs)


from . import transfer_guard_lib


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
    """Frontend wrapper for update.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.update(*args, **kwargs)


def update_thread_local_jit_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for update_thread_local_jit_state.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Result.
    """
    return _ops.update_thread_local_jit_state(*args, **kwargs)


def xla_runtime_errors(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager for `jax_experimental_unsafe_xla_runtime_errors` config option.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.xla_runtime_errors(*args, **kwargs)


import typing

import ml_switcheroo_compiler


def lib(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``lib``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
