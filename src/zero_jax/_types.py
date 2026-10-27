"""Canonical type definitions and aliases for zero-jax."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir import DType, LogicalGraph, LogicalMesh, PartitionSpec

R = TypeVar("R")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for array-like objects exposing shape and dtype."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array.

        Returns:
            tuple[int, ...]: The shape tuple.
        """
        ...

    @property
    def dtype(self) -> DType | str:
        """Return the data type of the array.

        Returns:
            DType | str: The data type.
        """
        ...


@runtime_checkable
class ToListProtocol(Protocol):
    """Protocol for serialized or custom array objects exposing ``tolist()``, ``shape``, and ``dtype``."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array.

        Returns:
            tuple[int, ...]: The shape tuple.
        """
        ...

    @property
    def dtype(self) -> DType | str:
        """Return the data type of the array.

        Returns:
            DType | str: The data type.
        """
        ...

    def tolist(self) -> PyTree:
        """Return the nested Python list or scalar representation of the array.

        Returns:
            PyTree: Nested Python list or scalar value.
        """
        ...


@runtime_checkable
class MappableDataclassProtocol(Protocol):
    """Protocol for mappable dataclasses exposing ``__chex_mappable_fields__`` and ``__chex_field_values__``."""

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of field names for PyTree traversal.

        Returns:
            tuple[str, ...]: Ordered field names.
        """
        ...

    def __chex_field_values__(self) -> tuple[PyTree, ...]:
        """Return the ordered tuple of field values matching ``__chex_mappable_fields__``.

        Returns:
            tuple[PyTree, ...]: Ordered field values.
        """
        ...


@runtime_checkable
class CustomStateMappingProtocol(Protocol):
    """Protocol for custom state mappings such as ``zero_flax.nnx.State`` and ``zero_flax.core.FrozenDict``."""

    def keys(self) -> Iterable[str]:
        """Return an iterable of mapping keys.

        Returns:
            Iterable[str]: Mapping keys.
        """
        ...

    def __getitem__(self, key: str) -> PyTree:
        """Retrieve the entry associated with ``key``.

        Args:
            key: Mapping key string.

        Returns:
            PyTree: Associated value.
        """
        ...


@runtime_checkable
class CustomVariableProtocol(Protocol):
    """Protocol for variable containers such as ``zero_flax.nnx.Variable`` preserving metadata across PyTree ops."""

    value: PyTree
    metadata: Mapping[str, ConfigValue]

    @property
    def var_type(self) -> str:
        """Return the variable container class name.

        Returns:
            str: Variable type name.
        """
        ...

    def replace(self, value: PyTree = None, **metadata: ConfigValue) -> CustomVariableProtocol:
        """Return a new variable container of the same type with an updated value and metadata.

        Args:
            value: Updated inner value.
            **metadata: Additional or updated metadata key-value pairs.

        Returns:
            CustomVariableProtocol: Updated variable instance.
        """
        ...


@runtime_checkable
class RecordProtocol(Protocol):
    """Protocol for dataset record containers pairing ``metadata`` with payload ``data``."""

    metadata: ConfigValue
    data: PyTree


@runtime_checkable
class StatefulProtocol(Protocol):
    """Protocol for stateful module objects with a state dictionary."""

    state: Mapping[str, Tensor | ArrayProtocol]


@runtime_checkable
class NamedProtocol(Protocol):
    """Protocol for dtype-like objects exposing a string name attribute."""

    name: str


class VariadicCallable(Protocol[T_co]):
    """Protocol for variadic functions accepting arbitrary arguments and returning ``T_co``."""

    __name__: str

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> T_co:
        """Invoke the variadic callable.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            T_co: Result value.
        """
        ...


if typing.TYPE_CHECKING:
    NumericScalar = int | float | complex | bool
    RealScalar = int | float | bool
    ScalarNumeric = int | float | complex | bool
    ShapeTuple = tuple[int, ...]
    ShapeSpec = int | Sequence[int]
    AxisIntOrTuple = int | Sequence[int] | None
    AxisSpec = int | Sequence[int] | None
    DTypeName = str
    DTypeLike = DType | str | type[bool | int | float | complex] | NamedProtocol | None
    DTypeSpec = DType | str | type[bool | int | float | complex] | NamedProtocol | None
    PaddingSpec = str | Sequence[tuple[int, int]] | Sequence[tuple[int, int, int]]
    PrecisionType = str | int | tuple[str, str] | tuple[int, int] | None
    SliceSpec = slice[int | float | None, int | float | None, int | float | None]
    IndexElement = int | SliceSpec | types.EllipsisType | Tensor | ArrayProtocol | None
    IndexSpec = IndexElement | tuple[IndexElement, ...]
    ArrayInput = Tensor | ArrayProtocol | ScalarNumeric | Sequence[ScalarNumeric] | Sequence[Sequence[ScalarNumeric]]
    OptionalArrayInput = ArrayInput | None
    BoolConfigValue = bool | None
    StringConfigValue = str | None
    IntConfigValue = int | None
    EnumConfigValue = str | int | None
    OptionalStringConfig = str | None
    ConfigValue = bool | int | float | str | ShapeTuple | DType | TensorOrScalar | ArrayProtocol | None
    PrimitiveParamValue = bool | int | float | str | ShapeTuple | DTypeSpec | None
    ArrayLike = TensorOrScalar | ArrayProtocol
    OptionalArrayLike = ArrayLike | None
    ArrayLikeSpec = TensorOrScalar | ArrayProtocol | None
    PyTreeLeaf = TensorOrScalar | ArrayProtocol | None
    PyTree = PyTreeLeaf | Sequence["PyTree"] | Mapping[str, "PyTree"]
else:
    NumericScalar = TypeVar("NumericScalar", int, float, complex, bool)
    RealScalar = TypeVar("RealScalar", int, float, bool)
    ScalarNumeric = TypeVar("ScalarNumeric", int, float, complex, bool)
    ShapeTuple = tuple[int, ...]
    ShapeSpec = TypeVar("ShapeSpec")
    AxisIntOrTuple = TypeVar("AxisIntOrTuple")
    AxisSpec = TypeVar("AxisSpec")
    DTypeName = str
    DTypeLike = TypeVar("DTypeLike")
    DTypeSpec = TypeVar("DTypeSpec")
    PaddingSpec = TypeVar("PaddingSpec")
    PrecisionType = TypeVar("PrecisionType")
    SliceSpec = slice
    IndexElement = TypeVar("IndexElement")
    IndexSpec = TypeVar("IndexSpec")
    ArrayInput = TypeVar("ArrayInput")
    OptionalArrayInput = TypeVar("OptionalArrayInput")
    BoolConfigValue = TypeVar("BoolConfigValue")
    StringConfigValue = TypeVar("StringConfigValue")
    IntConfigValue = TypeVar("IntConfigValue")
    EnumConfigValue = TypeVar("EnumConfigValue")
    OptionalStringConfig = TypeVar("OptionalStringConfig")
    ConfigValue = TypeVar("ConfigValue")
    PrimitiveParamValue = TypeVar("PrimitiveParamValue")
    ArrayLike = TypeVar("ArrayLike")
    OptionalArrayLike = TypeVar("OptionalArrayLike")
    ArrayLikeSpec = TypeVar("ArrayLikeSpec")
    PyTreeLeaf = TypeVar("PyTreeLeaf")
    PyTree = TypeVar("PyTree")

__all__ = [
    "ArrayInput",
    "ArrayLike",
    "ArrayLikeSpec",
    "ArrayProtocol",
    "AxisIntOrTuple",
    "AxisSpec",
    "BoolConfigValue",
    "Callable",
    "ConfigValue",
    "CustomStateMappingProtocol",
    "CustomVariableProtocol",
    "DType",
    "DTypeLike",
    "DTypeName",
    "DTypeSpec",
    "EnumConfigValue",
    "IndexElement",
    "IndexSpec",
    "IntConfigValue",
    "Iterable",
    "LogicalGraph",
    "LogicalMesh",
    "MappableDataclassProtocol",
    "NamedProtocol",
    "NumericScalar",
    "OptionalArrayInput",
    "OptionalArrayLike",
    "OptionalStringConfig",
    "PaddingSpec",
    "PartitionSpec",
    "PrecisionType",
    "PrimitiveParamValue",
    "PyTree",
    "PyTreeLeaf",
    "R",
    "RealScalar",
    "RecordProtocol",
    "ScalarNumeric",
    "ShapeSpec",
    "ShapeTuple",
    "SliceSpec",
    "StatefulProtocol",
    "StringConfigValue",
    "T",
    "T_co",
    "Tensor",
    "TensorOrScalar",
    "ToListProtocol",
    "VariadicCallable",
    "types",
    "typing",
]
