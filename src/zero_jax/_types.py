"""Canonical type definitions and aliases for zero-jax."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Mapping, Sequence
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
    "DType",
    "DTypeLike",
    "DTypeName",
    "DTypeSpec",
    "EnumConfigValue",
    "IndexElement",
    "IndexSpec",
    "IntConfigValue",
    "LogicalGraph",
    "LogicalMesh",
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
    "VariadicCallable",
    "types",
    "typing",
]
