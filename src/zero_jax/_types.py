"""Canonical type definitions and aliases for zero-jax."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeVar, Union, runtime_checkable

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
    from typing import ParamSpec, TypeAlias

    P = ParamSpec("P")
    NumericScalar: TypeAlias = int | float | complex | bool
    RealScalar: TypeAlias = int | float | bool
    ScalarNumeric: TypeAlias = NumericScalar
    ShapeTuple: TypeAlias = tuple[int, ...]
    ShapeSpec: TypeAlias = int | Sequence[int]
    AxisIntOrTuple: TypeAlias = int | Sequence[int] | None
    AxisSpec: TypeAlias = AxisIntOrTuple
    DTypeLike: TypeAlias = DType | str | type[int | float | complex | bool] | None
    DTypeSpec: TypeAlias = DTypeLike
    PaddingSpec: TypeAlias = str | Sequence[tuple[int, int]] | Sequence[tuple[int, int, int]]
    PrecisionType: TypeAlias = str | int | tuple[str, str] | tuple[int, int] | None
    IndexElement: TypeAlias = int | slice[int | Tensor | ArrayProtocol | None, int | Tensor | ArrayProtocol | None, int | Tensor | ArrayProtocol | None] | types.EllipsisType | Tensor | ArrayProtocol | None
    IndexSpec: TypeAlias = IndexElement | tuple[IndexElement, ...]
    ArrayInput: TypeAlias = TensorOrScalar | ArrayProtocol | Sequence[NumericScalar]
    OptionalArrayInput: TypeAlias = ArrayInput | None
    BoolConfigValue: TypeAlias = bool | None
    StringConfigValue: TypeAlias = str | None
    IntConfigValue: TypeAlias = int | None
    EnumConfigValue: TypeAlias = str | int | None
    OptionalStringConfig: TypeAlias = str | None
    ConfigValue: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PrimitiveParamValue: TypeAlias = bool | int | float | str | ShapeTuple | DTypeSpec | None
    ArrayLikeSpec: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PyTreeLeaf: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PyTree: TypeAlias = TensorOrScalar | ArrayProtocol | None
else:
    P = TypeVar("P")
    NumericScalar = TypeVar("NumericScalar")
    RealScalar = TypeVar("RealScalar")
    ScalarNumeric = NumericScalar
    ShapeTuple = tuple[int, ...]
    ShapeSpec = TypeVar("ShapeSpec")
    AxisIntOrTuple = TypeVar("AxisIntOrTuple")
    AxisSpec = AxisIntOrTuple
    DTypeLike = TypeVar("DTypeLike")
    DTypeSpec = DTypeLike
    PaddingSpec = TypeVar("PaddingSpec")
    PrecisionType = TypeVar("PrecisionType")
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
    ArrayLikeSpec = TypeVar("ArrayLikeSpec")
    PyTreeLeaf = TypeVar("PyTreeLeaf")
    PyTree = TypeVar("PyTree")

__all__ = [
    "ArrayInput",
    "ArrayLikeSpec",
    "ArrayProtocol",
    "AxisIntOrTuple",
    "AxisSpec",
    "BoolConfigValue",
    "Callable",
    "ConfigValue",
    "DType",
    "DTypeLike",
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
    "OptionalStringConfig",
    "P",
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
    "StatefulProtocol",
    "StringConfigValue",
    "T",
    "T_co",
    "Tensor",
    "TensorOrScalar",
    "VariadicCallable",
]
