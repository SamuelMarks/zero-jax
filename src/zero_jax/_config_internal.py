"""Canonical internal JAX configuration state and helpers."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from ml_switcheroo_compiler.core.config import config as _compiler_config

from zero_jax._types import ConfigValue


class FlagHolder:
    """Holder for a typed configuration flag."""

    def __init__(
        self,
        name: str = "flag",
        default: ConfigValue = None,
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
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``Config`` with default JAX configuration flags.

        Args:
            id: Optional identifier metadata.
            name: Optional name metadata.
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
            _compiler_config.default_float_dtype = "float64" if bool(val) else "float32"

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


config = Config()
flags = config


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
    "bool_env",
    "config",
    "default_device",
    "default_matmul_precision",
    "enable_x64",
    "flags",
    "int_env",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "str_env",
    "transfer_guard",
]
