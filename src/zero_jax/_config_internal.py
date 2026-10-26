"""Canonical internal JAX configuration state and helpers."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_ir import DType

from zero_jax._types import ArrayLikeSpec, ConfigValue


class FlagHolder:
    """Holder for a typed configuration flag."""

    def __init__(
        self,
        name: str = "flag",
        default: ConfigValue = None,
        *,
        id: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``FlagHolder``.

        Args:
            name: Flag name string.
            default: Default flag value.
            id: Optional identifier metadata.
            value: Explicit initial value override.
        """
        self.name = name
        self.default = default
        self.id = id
        self.value = value if value is not None else default


class Config:
    """Global and thread-local JAX configuration manager."""

    def __init__(
        self,
        *,
        id: ConfigValue = None,
        name: str = "jax_config",
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``Config`` with default JAX configuration flags.

        Args:
            id: Optional identifier metadata.
            name: Configuration manager name.
            value: Optional value metadata.
        """
        self.id = id
        self.name = name
        self.value = value
        self.values: dict[str, ConfigValue] = {
            "jax_enable_x64": False,
            "jax_default_matmul_precision": "highest",
            "jax_default_device": None,
            "jax_numpy_rank_promotion": "allow",
            "jax_numpy_dtype_promotion": "standard",
            "jax_transfer_guard": "allow",
            "jax_debug_nans": False,
            "jax_debug_infs": False,
            "jax_check_tracer_leaks": False,
            "jax_disable_jit": False,
        }

    def update(self, name: str, val: ConfigValue) -> None:
        """Update the value of configuration option ``name``.

        Args:
            name: Configuration option name.
            val: New value to assign.
        """
        key = name if name.startswith("jax_") else f"jax_{name}"
        self.values[key] = val
        self.values[name] = val
        if key == "jax_enable_x64":
            enabled = bool(val)
            _compiler_config.default_float_dtype = DType.Float64 if enabled else DType.Float32
            _compiler_config.default_int_dtype = DType.Int64 if enabled else DType.Int32

    def read(self, name: str) -> ConfigValue:
        """Read the current value of configuration option ``name``.

        Args:
            name: Configuration option name.

        Returns:
            ConfigValue: Current configuration setting.
        """
        key = name if name.startswith("jax_") else f"jax_{name}"
        return self.values.get(key, self.values.get(name))

    @property
    def jax_enable_x64(self) -> bool:
        """Return whether 64-bit precision is enabled.

        Returns:
            bool: ``True`` if x64 mode is active.
        """
        return bool(self.values.get("jax_enable_x64", False))

    def __call__(self) -> Config:
        """Return this :class:`Config` instance when invoked as a callable.

        Returns:
            Config: This configuration manager instance.
        """
        return self


config = Config()
flags = config


class JitState:
    """Global or thread-local JIT execution state."""

    def __init__(
        self,
        disable_jit: bool = False,
        enable_x64: bool = False,
        default_device: str | None = None,
        extra_jit_context: ConfigValue = None,
    ) -> None:
        """Initialize ``JitState``.

        Args:
            disable_jit: Whether JIT compilation is disabled.
            enable_x64: Whether 64-bit precision is enabled.
            default_device: Default execution device name.
            extra_jit_context: Optional extra JIT cache key context.
        """
        self.disable_jit: bool = bool(disable_jit)
        self.enable_x64: bool = bool(enable_x64)
        self.default_device: str | None = default_device
        self.extra_jit_context: ConfigValue = extra_jit_context


class PyArgSignature:
    """Static argument signature descriptor for JIT cache lookup."""

    def __init__(
        self,
        dtype: str = "float32",
        shape: tuple[int, ...] = (),
        weak_type: bool = False,
    ) -> None:
        """Initialize ``PyArgSignature``.

        Args:
            dtype: Element dtype string.
            shape: Static shape tuple.
            weak_type: Whether the argument has a weak scalar type.
        """
        self.dtype: str = str(dtype)
        self.shape: tuple[int, ...] = tuple(shape)
        self.weak_type: bool = bool(weak_type)


_GLOBAL_JIT_STATE = JitState()
_THREAD_LOCAL_JIT_STATE = JitState()
_THREAD_LOCAL_INIT_CALLBACK: ArrayLikeSpec = None


def get_enable_x64() -> bool:
    """Return whether 64-bit mode is enabled.

    Returns:
        bool: ``True`` when ``jax_enable_x64`` is active.
    """
    return config.jax_enable_x64


def global_jit_state() -> JitState:
    """Return the global :class:`JitState` instance.

    Returns:
        JitState: Global JIT state object.
    """
    _GLOBAL_JIT_STATE.enable_x64 = config.jax_enable_x64
    return _GLOBAL_JIT_STATE


def thread_local_jit_state() -> JitState:
    """Return the thread-local :class:`JitState` instance.

    Returns:
        JitState: Thread-local JIT state object.
    """
    return _THREAD_LOCAL_JIT_STATE


def set_thread_local_state_initialization_callback(callback: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Register a callback invoked when thread-local JIT state is initialized.

    Args:
        callback: Initialization callable or ``None``.

    Returns:
        ArrayLikeSpec: Registered callback.
    """
    global _THREAD_LOCAL_INIT_CALLBACK
    _THREAD_LOCAL_INIT_CALLBACK = callback
    return callback


def swap_thread_local_state_disable_jit(val: bool | None = False) -> bool:
    """Swap the thread-local ``disable_jit`` flag and return its previous value.

    Args:
        val: New boolean state for ``disable_jit``.

    Returns:
        bool: Previous ``disable_jit`` state.
    """
    prev = _THREAD_LOCAL_JIT_STATE.disable_jit
    _THREAD_LOCAL_JIT_STATE.disable_jit = bool(val)
    return prev


_DEBUG_LOGGERS_ENABLED: dict[str, bool] = {}


def enable_debug_logging(logger_name: str = "jax") -> str:
    """Enable debug logging for ``logger_name``.

    Args:
        logger_name: Name of the logger to configure.

    Returns:
        str: Configured logger name.
    """
    _DEBUG_LOGGERS_ENABLED[logger_name] = True
    return logger_name


def disable_all_debug_logging() -> bool:
    """Disable all debug loggers previously enabled via :func:`enable_debug_logging`.

    Returns:
        bool: ``False`` indicating debug logging is disabled.
    """
    _DEBUG_LOGGERS_ENABLED.clear()
    return False


class TransferGuardLevel:
    """Enumeration of device transfer guard enforcement levels."""

    ALLOW: str = "allow"
    LOG: str = "log"
    DISALLOW: str = "disallow"
    LOG_EXPLICIT: str = "log_explicit"
    DISALLOW_EXPLICIT: str = "disallow_explicit"

    def __init__(self, level: str = "allow") -> None:
        """Initialize ``TransferGuardLevel``.

        Args:
            level: Guard level identifier string.
        """
        self.level: str = str(level)


class TransferGuardState:
    """Global or thread-local device transfer guard state."""

    def __init__(
        self,
        host_to_device: str = "allow",
        device_to_device: str = "allow",
        device_to_host: str = "allow",
        explicit_device_put: bool = False,
        explicit_device_get: bool = False,
    ) -> None:
        """Initialize ``TransferGuardState``.

        Args:
            host_to_device: Transfer guard level for host-to-device transfers.
            device_to_device: Transfer guard level for device-to-device transfers.
            device_to_host: Transfer guard level for device-to-host transfers.
            explicit_device_put: Whether inside an explicit ``device_put`` scope.
            explicit_device_get: Whether inside an explicit ``device_get`` scope.
        """
        self.host_to_device: str = host_to_device
        self.device_to_device: str = device_to_device
        self.device_to_host: str = device_to_host
        self.explicit_device_put: bool = bool(explicit_device_put)
        self.explicit_device_get: bool = bool(explicit_device_get)


_GLOBAL_TRANSFER_GUARD_STATE = TransferGuardState()
_THREAD_LOCAL_TRANSFER_GUARD_STATE = TransferGuardState()


def global_transfer_guard_state() -> TransferGuardState:
    """Return the global :class:`TransferGuardState`.

    Returns:
        TransferGuardState: Global transfer guard state.
    """
    return _GLOBAL_TRANSFER_GUARD_STATE


def thread_local_transfer_guard_state() -> TransferGuardState:
    """Return the thread-local :class:`TransferGuardState`.

    Returns:
        TransferGuardState: Thread-local transfer guard state.
    """
    return _THREAD_LOCAL_TRANSFER_GUARD_STATE


def bool_env(varname: str = "", default: bool = False) -> bool:
    """Read an environment variable and interpret it as a boolean.

    Args:
        varname: Environment variable name.
        default: Default value when unset.

    Returns:
        bool: Parsed boolean value.
    """
    if not varname or varname not in os.environ:
        return default
    val = os.environ[varname].strip().lower()
    return val in ("1", "true", "yes", "on")


def int_env(varname: str = "", default: int = 0) -> int:
    """Read an environment variable and interpret it as an integer.

    Args:
        varname: Environment variable name.
        default: Default integer value when unset.

    Returns:
        int: Parsed integer value.
    """
    if not varname or varname not in os.environ:
        return default
    return int(os.environ[varname])


def str_env(varname: str = "", default: str = "") -> str:
    """Read an environment variable as a string.

    Args:
        varname: Environment variable name.
        default: Default string value when unset.

    Returns:
        str: Environment variable string.
    """
    if not varname:
        return default
    return os.environ.get(varname, default)


@contextlib.contextmanager
def enable_x64(new_val: bool = True) -> Iterator[None]:
    """Context manager to temporarily enable or disable 64-bit precision.

    Args:
        new_val: Whether x64 precision should be enabled inside the block.

    Yields:
        None: Context with updated ``jax_enable_x64`` setting.
    """
    prev = config.jax_enable_x64
    config.update("jax_enable_x64", new_val)
    try:
        yield
    finally:
        config.update("jax_enable_x64", prev)


@contextlib.contextmanager
def default_device(device: ConfigValue = None) -> Iterator[None]:
    """Context manager to temporarily set the default device.

    Args:
        device: Target device for operations inside the block.

    Yields:
        None: Context with updated ``jax_default_device`` setting.
    """
    prev = config.read("jax_default_device")
    config.update("jax_default_device", device)
    try:
        yield
    finally:
        config.update("jax_default_device", prev)


@contextlib.contextmanager
def default_matmul_precision(precision: str = "highest") -> Iterator[None]:
    """Context manager to temporarily set the default matrix multiplication precision.

    Args:
        precision: Precision level string.

    Yields:
        None: Context with updated ``jax_default_matmul_precision`` setting.
    """
    prev = config.read("jax_default_matmul_precision")
    config.update("jax_default_matmul_precision", precision)
    try:
        yield
    finally:
        config.update("jax_default_matmul_precision", prev)


@contextlib.contextmanager
def numpy_dtype_promotion(mode: str = "standard") -> Iterator[None]:
    """Context manager to temporarily set the NumPy dtype promotion mode.

    Args:
        mode: Promotion mode (``"standard"`` or ``"strict"``).

    Yields:
        None: Context with updated ``jax_numpy_dtype_promotion`` setting.
    """
    prev = config.read("jax_numpy_dtype_promotion")
    config.update("jax_numpy_dtype_promotion", mode)
    try:
        yield
    finally:
        config.update("jax_numpy_dtype_promotion", prev)


@contextlib.contextmanager
def numpy_rank_promotion(mode: str = "allow") -> Iterator[None]:
    """Context manager to temporarily set the NumPy rank promotion mode.

    Args:
        mode: Rank promotion mode (``"allow"``, ``"warn"``, or ``"raise"``).

    Yields:
        None: Context with updated ``jax_numpy_rank_promotion`` setting.
    """
    prev = config.read("jax_numpy_rank_promotion")
    config.update("jax_numpy_rank_promotion", mode)
    try:
        yield
    finally:
        config.update("jax_numpy_rank_promotion", prev)


@contextlib.contextmanager
def transfer_guard(level: str = "allow") -> Iterator[None]:
    """Context manager to temporarily set the device transfer guard level.

    Args:
        level: Transfer guard level (``"allow"``, ``"log"``, ``"disallow"``).

    Yields:
        None: Context with updated ``jax_transfer_guard`` setting.
    """
    prev = config.read("jax_transfer_guard")
    config.update("jax_transfer_guard", level)
    try:
        yield
    finally:
        config.update("jax_transfer_guard", prev)


__all__ = [
    "Config",
    "FlagHolder",
    "JitState",
    "PyArgSignature",
    "TransferGuardLevel",
    "TransferGuardState",
    "bool_env",
    "config",
    "default_device",
    "default_matmul_precision",
    "disable_all_debug_logging",
    "enable_debug_logging",
    "enable_x64",
    "flags",
    "get_enable_x64",
    "global_jit_state",
    "global_transfer_guard_state",
    "int_env",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "set_thread_local_state_initialization_callback",
    "str_env",
    "swap_thread_local_state_disable_jit",
    "thread_local_jit_state",
    "thread_local_transfer_guard_state",
    "transfer_guard",
]
