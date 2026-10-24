"""Canonical type definitions and aliases for zero-jax."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir import DType

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

    from ml_switcheroo_compiler.core.shape import TensorOrScalar

    P = ParamSpec("P")
    NumericScalar: TypeAlias = int | float | complex | bool
    RealScalar: TypeAlias = int | float | bool
    ScalarNumeric: TypeAlias = NumericScalar
    ShapeTuple: TypeAlias = tuple[int, ...]
    ShapeSpec: TypeAlias = int | Sequence[int]
    AxisIntOrTuple: TypeAlias = int | Sequence[int] | None
    AxisSpec: TypeAlias = AxisIntOrTuple
    DTypeSpec: TypeAlias = DType | str | type[int | float | complex | bool] | None
    PaddingSpec: TypeAlias = str | Sequence[tuple[int, int]] | Sequence[tuple[int, int, int]]
    PrecisionType: TypeAlias = str | int | tuple[str, str] | tuple[int, int] | None
    IndexElement: TypeAlias = int | slice[int | Tensor | ArrayProtocol | None, int | Tensor | ArrayProtocol | None, int | Tensor | ArrayProtocol | None] | types.EllipsisType | Tensor | ArrayProtocol | None
    IndexSpec: TypeAlias = IndexElement | tuple[IndexElement, ...]
    ArrayInput: TypeAlias = TensorOrScalar | ArrayProtocol | Sequence[NumericScalar]
    OptionalArrayInput: TypeAlias = ArrayInput | None
    BoolConfigValue: TypeAlias = bool | None
    StringConfigValue: TypeAlias = str | None
    IntConfigValue: TypeAlias = int | None
    OptionalStringConfig: TypeAlias = str | None
    ConfigValue: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PrimitiveParamValue: TypeAlias = bool | int | float | str | ShapeTuple | DTypeSpec | None
    ArrayLikeSpec: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PyTreeLeaf: TypeAlias = TensorOrScalar | ArrayProtocol | None
    PyTree: TypeAlias = TensorOrScalar | ArrayProtocol | None
else:
    P = TypeVar("P")
    NumericScalar = "int | float | complex | bool"
    RealScalar = "int | float | bool"
    ScalarNumeric = "int | float | complex | bool"
    ShapeTuple = "tuple[int, ...]"
    ShapeSpec = "int | Sequence[int]"
    AxisIntOrTuple = "int | Sequence[int] | None"
    AxisSpec = "int | Sequence[int] | None"
    DTypeSpec = "DType | str | type[int | float | complex | bool] | None"
    PaddingSpec = "str | Sequence[tuple[int, int]] | Sequence[tuple[int, int, int]]"
    PrecisionType = "str | int | tuple[str, str] | tuple[int, int] | None"
    IndexElement = "int | slice | types.EllipsisType | None | Tensor | ArrayProtocol"
    IndexSpec = "IndexElement | tuple[IndexElement, ...]"
    ArrayInput = "ArrayProtocol | Tensor | NumericScalar | Sequence[ArrayInput]"
    OptionalArrayInput = "ArrayInput | None"
    BoolConfigValue = "bool | None"
    StringConfigValue = "str | None"
    IntConfigValue = "int | None"
    OptionalStringConfig = "str | None"
    ConfigValue = "bool | int | float | str | bytes | tuple[ConfigValue, ...] | list[ConfigValue] | dict[str, ConfigValue] | None"
    PrimitiveParamValue = "bool | int | float | str | ShapeTuple | DTypeSpec | None"
    ArrayLikeSpec = "ArrayProtocol | Tensor | ScalarNumeric | Sequence[ScalarNumeric] | None"
    PyTreeLeaf = "ArrayProtocol | Tensor | ScalarNumeric | str | bytes | None"
    PyTree = "PyTreeLeaf | tuple[PyTree, ...] | list[PyTree] | dict[str, PyTree]"

__all__ = [
    "ArrayInput",
    "ArrayLikeSpec",
    "ArrayProtocol",
    "AxisIntOrTuple",
    "AxisSpec",
    "BoolConfigValue",
    "ConfigValue",
    "DTypeSpec",
    "IndexElement",
    "IndexSpec",
    "IntConfigValue",
    "NamedProtocol",
    "NumericScalar",
    "OptionalArrayInput",
    "OptionalStringConfig",
    "P",
    "PaddingSpec",
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
    "VariadicCallable",
]
