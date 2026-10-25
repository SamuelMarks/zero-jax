"""Export submodule for zero_jax."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


class DisabledSafetyCheck:
    """Safety check configuration that can be disabled during export."""

    def __init__(self, kind: str = "custom", value: str = "") -> None:
        """Initialize ``DisabledSafetyCheck``.

        Args:
            kind: Safety check category.
            value: Target name or platform.
        """
        self.kind = kind
        self.value = value

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Construct a ``DisabledSafetyCheck`` instance.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return cls()


class Exported:
    """Exported JAX function representation."""

    def __init__(self, fun_name: str = "exported", platforms: tuple[str, ...] = ("cpu",), fun: ConfigValue = None) -> None:
        """Initialize ``Exported``.

        Args:
            fun_name: Name of the exported function.
            platforms: Tuple of lowering platforms.
            fun: Optional underlying callable.
        """
        self.fun_name = fun_name
        self.platforms = platforms
        self._fun = fun

    def call(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the exported function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Execution result.
        """
        if callable(self._fun):
            return self._fun(*args, **kwargs)
        return None

    def serialize(self, vjp_order: int = 0) -> bytearray:
        """Serialize the exported function to a byte array.

        Args:
            vjp_order: Maximum VJP order to include.

        Returns:
            bytearray: Serialized payload.
        """
        return bytearray(self.fun_name.encode("utf-8"))

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Construct an ``Exported`` instance.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return cls()


class SymbolicScope:
    """Scope for symbolic dimension expressions."""

    def __init__(self, constraints: tuple[str, ...] = ()) -> None:
        """Initialize ``SymbolicScope``.

        Args:
            constraints: Tuple of symbolic dimension constraint strings.
        """
        self.constraints = tuple(constraints)

    @classmethod
    def apply(cls, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Construct a ``SymbolicScope`` instance.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.

        Returns:
            The operation result.
        """
        return cls()


def default_export_platform(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the default export platform string.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return "cpu"


def deserialize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Deserialize an :class:`Exported` payload.

    Args:
        *args: Input parameter ``args`` (``payload``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and isinstance(args[0], (bytes, bytearray)):
        return Exported(fun_name=bytes(args[0]).decode("utf-8", errors="ignore") or "exported")
    return Exported()


def export(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Export a JAX function to an :class:`Exported` producer callable.

    Args:
        *args: Input parameter ``args`` (``fun``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    fun = args[0] if args else None

    def _produce(*trace_args: ArrayLikeSpec, **trace_kwargs: ArrayLikeSpec) -> Exported:
        """Trace ``fun`` on ``trace_args`` and return an :class:`Exported` object.

        Args:
            *trace_args: Example positional arguments.
            **trace_kwargs: Example keyword arguments.

        Returns:
            Exported: Exported representation.
        """
        return Exported(fun=fun)

    return _produce if callable(fun) else Exported()


def is_symbolic_dim(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return ``True`` if ``d`` is a symbolic dimension expression.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        return isinstance(args[0], str)
    return False


def maximum_supported_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the maximum supported export calling convention version.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return 9


def minimum_supported_calling_convention_version(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the minimum supported export calling convention version.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return 9


def symbolic_args_specs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct a PyTree of ``ShapeDtypeStruct`` specs with symbolic dimensions.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return args[0] if args else None


def symbolic_shape(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Parse a comma-separated symbolic shape specification into a tuple of dimensions.

    Args:
        *args: Input parameter ``args`` (``shape_spec``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and isinstance(args[0], str):
        parts = [p.strip() for p in args[0].split(",") if p.strip()]
        return tuple(int(p) if p.isdigit() else p for p in parts)
    return ()


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
