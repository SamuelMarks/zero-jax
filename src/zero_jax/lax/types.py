"""Data types and configs for jax.lax."""

from __future__ import annotations

from typing import Any


class ConvDimensionNumbers:
    def __init__(self, lhs_spec: Any, rhs_spec: Any, out_spec: Any) -> None:
        self.lhs_spec = lhs_spec
        self.rhs_spec = rhs_spec
        self.out_spec = out_spec


class ConvGeneralDilatedDimensionNumbers:
    def __init__(self, lhs_spec: Any, rhs_spec: Any, out_spec: Any) -> None:
        self.lhs_spec = lhs_spec
        self.rhs_spec = rhs_spec
        self.out_spec = out_spec


class DotDimensionNumbers:
    def __init__(
        self, lhs_contracting: Any, rhs_contracting: Any, lhs_batch: Any, rhs_batch: Any
    ) -> None:
        self.lhs_contracting = lhs_contracting
        self.rhs_contracting = rhs_contracting
        self.lhs_batch = lhs_batch
        self.rhs_batch = rhs_batch


class GatherDimensionNumbers:
    def __init__(
        self, offset_dims: Any, collapsed_slice_dims: Any, start_index_map: Any
    ) -> None:
        self.offset_dims = offset_dims
        self.collapsed_slice_dims = collapsed_slice_dims
        self.start_index_map = start_index_map


class GatherScatterMode:
    """Enum equivalent for scatter/gather mode."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Precision:
    """Precision enum."""

    DEFAULT = "DEFAULT"
    HIGH = "HIGH"
    HIGHEST = "HIGHEST"


class PrecisionLike:
    """Type alias representation."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class RandomAlgorithm:
    """Random algorithm enum."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class RoundingMethod:
    """Rounding method enum."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ScatterDimensionNumbers:
    def __init__(
        self,
        update_window_dims: Any,
        inserted_window_dims: Any,
        scatter_dims_to_operand_dims: Any,
    ) -> None:
        self.update_window_dims = update_window_dims
        self.inserted_window_dims = inserted_window_dims
        self.scatter_dims_to_operand_dims = scatter_dims_to_operand_dims
