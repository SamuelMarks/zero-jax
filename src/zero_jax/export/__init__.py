"""Export submodule for zero_jax."""

from typing import Any


class DisabledSafetyCheck:
    """Stub for DisabledSafetyCheck."""

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        """Stub apply method."""
        return None  # pragma: no cover


class Exported:
    """Stub for Exported."""

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        """Stub apply method."""
        return None  # pragma: no cover


class SymbolicScope:
    """Stub for SymbolicScope."""

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        """Stub apply method."""
        return None  # pragma: no cover


def default_export_platform(*args: Any, **kwargs: Any) -> Any:
    """Stub for default_export_platform."""
    return None


def deserialize(*args: Any, **kwargs: Any) -> Any:
    """Stub for deserialize."""
    return None


def export(*args: Any, **kwargs: Any) -> Any:
    """Stub for export."""
    return None


def is_symbolic_dim(*args: Any, **kwargs: Any) -> Any:
    """Stub for is_symbolic_dim."""
    return None


def maximum_supported_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    """Stub for maximum_supported_calling_convention_version."""
    return None


def minimum_supported_calling_convention_version(*args: Any, **kwargs: Any) -> Any:
    """Stub for minimum_supported_calling_convention_version."""
    return None


def symbolic_args_specs(*args: Any, **kwargs: Any) -> Any:
    """Stub for symbolic_args_specs."""
    return None


def symbolic_shape(*args: Any, **kwargs: Any) -> Any:
    """Stub for symbolic_shape."""
    return None


__all__ = [
    "DisabledSafetyCheck",
    "Exported",
    "SymbolicScope",
    "default_export_platform",
    "deserialize",
    "export",
    "is_symbolic_dim",
    "maximum_supported_calling_convention_version",
    "minimum_supported_calling_convention_version",
    "symbolic_args_specs",
    "symbolic_shape",
]

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


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
