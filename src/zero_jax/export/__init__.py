"""Export submodule for zero_jax."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


class DisabledSafetyCheck:
    """Stub for DisabledSafetyCheck."""

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Stub apply method.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return None


class Exported:
    """Stub for Exported."""

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Stub apply method.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return None


class SymbolicScope:
    """Stub for SymbolicScope."""

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Stub apply method.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return None


def default_export_platform(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for default_export_platform.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def deserialize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for deserialize.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def export(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for export.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def is_symbolic_dim(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for is_symbolic_dim.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def maximum_supported_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for maximum_supported_calling_convention_version.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def minimum_supported_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for minimum_supported_calling_convention_version.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def symbolic_args_specs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for symbolic_args_specs.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def symbolic_shape(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for symbolic_shape.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
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
