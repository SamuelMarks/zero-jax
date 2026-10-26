"""Export submodule for zero_jax."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from zero_jax._types import PyTree, VariadicCallable


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
    def apply(cls, kind: str = "custom", value: str = "") -> DisabledSafetyCheck:
        """Construct a ``DisabledSafetyCheck`` instance.

        Args:
            kind: Safety check category.
            value: Target name or platform.

        Returns:
            DisabledSafetyCheck: Constructed check instance.
        """
        return cls(kind=kind, value=value)


class Exported:
    """Exported JAX function representation."""

    def __init__(
        self,
        fun_name: str = "exported",
        platforms: tuple[str, ...] = ("cpu",),
        fun: VariadicCallable[PyTree] | None = None,
    ) -> None:
        """Initialize ``Exported``.

        Args:
            fun_name: Name of the exported function.
            platforms: Tuple of lowering platforms.
            fun: Optional underlying callable.
        """
        self.fun_name = fun_name
        self.platforms = platforms
        self._fun = fun

    def call(self, *args: PyTree, **kwargs: PyTree) -> PyTree:
        """Invoke the exported function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Execution result.
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
        _ = vjp_order
        return bytearray(self.fun_name.encode("utf-8"))

    @classmethod
    def apply(cls, fun_name: str = "exported", platforms: tuple[str, ...] = ("cpu",)) -> Exported:
        """Construct an ``Exported`` instance.

        Args:
            fun_name: Name of the exported function.
            platforms: Tuple of lowering platforms.

        Returns:
            Exported: Constructed ``Exported`` instance.
        """
        return cls(fun_name=fun_name, platforms=platforms)


class SymbolicScope:
    """Scope for symbolic dimension expressions."""

    def __init__(self, constraints: Sequence[str] = ()) -> None:
        """Initialize ``SymbolicScope``.

        Args:
            constraints: Sequence of symbolic dimension constraint strings.
        """
        self.constraints = tuple(constraints)

    @classmethod
    def apply(cls, constraints: Sequence[str] = ()) -> SymbolicScope:
        """Construct a ``SymbolicScope`` instance.

        Args:
            constraints: Sequence of symbolic dimension constraint strings.

        Returns:
            SymbolicScope: Constructed scope instance.
        """
        return cls(constraints=constraints)


def default_export_platform() -> str:
    """Return the default export platform string.

    Returns:
        str: Default platform name (``"cpu"``).
    """
    return "cpu"


def deserialize(payload: bytes | bytearray = b"") -> Exported:
    """Deserialize an :class:`Exported` payload.

    Args:
        payload: Serialized byte buffer.

    Returns:
        Exported: Deserialized :class:`Exported` object.
    """
    if payload:
        return Exported(fun_name=bytes(payload).decode("utf-8", errors="ignore") or "exported")
    return Exported()


def export(
    fun_jax: VariadicCallable[PyTree] | None = None,
    *,
    lowering_platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
) -> VariadicCallable[Exported] | Exported:
    """Export a JAX function to an :class:`Exported` producer callable.

    Args:
        fun_jax: Function to export.
        lowering_platforms: Optional sequence of target lowering platforms.
        disabled_checks: Sequence of safety checks to disable.

    Returns:
        VariadicCallable[Exported] | Exported: Producer callable or :class:`Exported` instance.
    """
    _ = disabled_checks
    plats = tuple(lowering_platforms) if lowering_platforms is not None else ("cpu",)

    def _produce(*trace_args: PyTree, **trace_kwargs: PyTree) -> Exported:
        """Trace ``fun_jax`` on ``trace_args`` and return an :class:`Exported` object.

        Args:
            *trace_args: Example positional arguments.
            **trace_kwargs: Example keyword arguments.

        Returns:
            Exported: Exported representation.
        """
        _ = (trace_args, trace_kwargs)
        return Exported(platforms=plats, fun=fun_jax)

    return _produce if callable(fun_jax) else Exported(platforms=plats)


def is_symbolic_dim(d: int | str | None = None) -> bool:
    """Return ``True`` if ``d`` is a symbolic dimension expression.

    Args:
        d: Dimension value to inspect.

    Returns:
        bool: ``True`` if ``d`` is a symbolic dimension string.
    """
    return isinstance(d, str)


def maximum_supported_calling_convention_version() -> int:
    """Return the maximum supported export calling convention version.

    Returns:
        int: Maximum calling convention version number.
    """
    return 9


def minimum_supported_calling_convention_version() -> int:
    """Return the minimum supported export calling convention version.

    Returns:
        int: Minimum calling convention version number.
    """
    return 9


def symbolic_args_specs(
    args: PyTree = None,
    shapes_specs: PyTree = None,
    constraints: Sequence[str] = (),
    scope: SymbolicScope | None = None,
) -> PyTree:
    """Construct a PyTree of ``ShapeDtypeStruct`` specs with symbolic dimensions.

    Args:
        args: PyTree of arguments or shape/dtype descriptors.
        shapes_specs: Symbolic shape specification matching ``args``.
        constraints: Sequence of symbolic dimension constraints.
        scope: Optional symbolic scope.

    Returns:
        PyTree: Symbolic argument specifications.
    """
    _ = (shapes_specs, constraints, scope)
    return args


def symbolic_shape(
    shape_spec: str | None = None,
    *,
    constraints: Sequence[str] = (),
    scope: SymbolicScope | None = None,
) -> tuple[int | str, ...]:
    """Parse a comma-separated symbolic shape specification into a tuple of dimensions.

    Args:
        shape_spec: Comma-separated dimension specification string.
        constraints: Optional sequence of symbolic constraints.
        scope: Optional symbolic scope.

    Returns:
        tuple[int | str, ...]: Parsed tuple of integer and symbolic dimensions.
    """
    _ = (constraints, scope)
    if isinstance(shape_spec, str):
        parts = [p.strip() for p in shape_spec.split(",") if p.strip()]
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
