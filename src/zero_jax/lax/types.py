"""Data types and configs for jax.lax."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


class ConvDimensionNumbers:
    """Dimension layout specification for ``lax.conv_general_dilated``."""

    def __init__(self, lhs_spec: ArrayLikeSpec, rhs_spec: ArrayLikeSpec, out_spec: ArrayLikeSpec) -> None:
        """Initialize ``ConvDimensionNumbers``.

        Args:
            lhs_spec: Input tensor dimension layout tuple.
            rhs_spec: Kernel tensor dimension layout tuple.
            out_spec: Output tensor dimension layout tuple.
        """
        self.lhs_spec = lhs_spec
        self.rhs_spec = rhs_spec
        self.out_spec = out_spec


class ConvGeneralDilatedDimensionNumbers:
    """General dilated convolution dimension specification."""

    def __init__(self, lhs_spec: ArrayLikeSpec, rhs_spec: ArrayLikeSpec, out_spec: ArrayLikeSpec) -> None:
        """Initialize ``ConvGeneralDilatedDimensionNumbers``.

        Args:
            lhs_spec: Input tensor dimension layout specification.
            rhs_spec: Kernel tensor dimension layout specification.
            out_spec: Output tensor dimension layout specification.
        """
        self.lhs_spec = lhs_spec
        self.rhs_spec = rhs_spec
        self.out_spec = out_spec


class DotDimensionNumbers:
    """Dimension specification for ``lax.dot_general``."""

    def __init__(
        self,
        lhs_contracting: ArrayLikeSpec,
        rhs_contracting: ArrayLikeSpec,
        lhs_batch: ArrayLikeSpec,
        rhs_batch: ArrayLikeSpec,
    ) -> None:
        """Initialize ``DotDimensionNumbers``.

        Args:
            lhs_contracting: Left-hand contracting dimension indices.
            rhs_contracting: Right-hand contracting dimension indices.
            lhs_batch: Left-hand batch dimension indices.
            rhs_batch: Right-hand batch dimension indices.
        """
        self.lhs_contracting = lhs_contracting
        self.rhs_contracting = rhs_contracting
        self.lhs_batch = lhs_batch
        self.rhs_batch = rhs_batch


class GatherDimensionNumbers:
    """Dimension specification for ``lax.gather``."""

    def __init__(
        self,
        offset_dims: ArrayLikeSpec,
        collapsed_slice_dims: ArrayLikeSpec,
        start_index_map: ArrayLikeSpec,
    ) -> None:
        """Initialize ``GatherDimensionNumbers``.

        Args:
            offset_dims: Dimensions in the output that are offset into slices.
            collapsed_slice_dims: Dimensions in the operand that are collapsed.
            start_index_map: Mapping from index vector elements to operand dimensions.
        """
        self.offset_dims = offset_dims
        self.collapsed_slice_dims = collapsed_slice_dims
        self.start_index_map = start_index_map


class GatherScatterMode:
    """Out-of-bounds indexing behavior mode for ``lax.gather`` and ``lax.scatter``."""

    PROMISE_IN_BOUNDS = "PROMISE_IN_BOUNDS"
    CLIP = "CLIP"
    FILL_OR_DROP = "FILL_OR_DROP"
    ONE_HOT = "ONE_HOT"

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        dummy: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``GatherScatterMode``.

        Args:
            *args: Optional positional mode arguments.
            id: Optional identifier metadata.
            name: Optional mode name.
            value: Optional mode value (defaults to ``"FILL_OR_DROP"``).
            dummy: Optional legacy test parameter.
            **kwargs: Optional additional keyword parameters.
        """
        resolved = args[0] if args else (value if value is not None else self.FILL_OR_DROP)
        self.args = args
        self.id = id
        self.name = str(name) if name is not None else str(resolved)
        self.value = resolved
        self.dummy = dummy
        self.kwargs = kwargs

    @classmethod
    def from_any(cls, mode: str | GatherScatterMode | None) -> GatherScatterMode:
        """Normalize a string or ``GatherScatterMode`` into a ``GatherScatterMode`` instance.

        Args:
            mode: Input mode specification.

        Returns:
            GatherScatterMode: Normalized mode instance.
        """
        if isinstance(mode, GatherScatterMode):
            return mode
        if isinstance(mode, str):
            return cls(value=mode.upper(), name=mode.upper())
        return cls(value=cls.FILL_OR_DROP, name=cls.FILL_OR_DROP)


class Precision:
    """Matrix multiplication and convolution precision levels for ``jax.lax``."""

    DEFAULT = "DEFAULT"
    HIGH = "HIGH"
    HIGHEST = "HIGHEST"


class PrecisionLike:
    """Representation of a precision specification accepted by ``lax`` operations."""

    DEFAULT = Precision.DEFAULT
    HIGH = Precision.HIGH
    HIGHEST = Precision.HIGHEST

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        dummy: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``PrecisionLike``.

        Args:
            *args: Optional positional precision values.
            id: Optional identifier metadata.
            name: Optional precision name.
            value: Optional precision value.
            dummy: Optional legacy test parameter.
            **kwargs: Optional additional keyword parameters.
        """
        resolved = args[0] if args else (value if value is not None else Precision.DEFAULT)
        self.args = args
        self.id = id
        self.name = str(name) if name is not None else str(resolved)
        self.value = resolved
        self.dummy = dummy
        self.kwargs = kwargs


class RandomAlgorithm:
    """PRNG bit generation algorithm identifiers for ``lax.rng_bit_generator``."""

    RNG_DEFAULT = "RNG_DEFAULT"
    RNG_THREE_FRY = "RNG_THREE_FRY"
    RNG_PHILOX = "RNG_PHILOX"

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        dummy: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``RandomAlgorithm``.

        Args:
            *args: Optional positional algorithm identifiers.
            id: Optional identifier metadata.
            name: Optional algorithm name.
            value: Optional algorithm value.
            dummy: Optional legacy test parameter.
            **kwargs: Optional additional keyword parameters.
        """
        resolved = args[0] if args else (value if value is not None else self.RNG_DEFAULT)
        self.args = args
        self.id = id
        self.name = str(name) if name is not None else str(resolved)
        self.value = resolved
        self.dummy = dummy
        self.kwargs = kwargs


class RoundingMethod:
    """Rounding tie-breaking method for ``lax.round``."""

    AWAY_FROM_ZERO = 0
    TO_NEAREST_EVEN = 1

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        dummy: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``RoundingMethod``.

        Args:
            *args: Optional positional rounding method values.
            id: Optional identifier metadata.
            name: Optional rounding method name.
            value: Optional rounding method integer code.
            dummy: Optional legacy test parameter.
            **kwargs: Optional additional keyword parameters.
        """
        resolved = args[0] if args else (value if value is not None else self.TO_NEAREST_EVEN)
        self.args = args
        self.id = id
        self.name = str(name) if name is not None else str(resolved)
        self.value = resolved
        self.dummy = dummy
        self.kwargs = kwargs


class ScatterDimensionNumbers:
    """Dimension specification for ``lax.scatter``."""

    def __init__(
        self,
        update_window_dims: ArrayLikeSpec,
        inserted_window_dims: ArrayLikeSpec,
        scatter_dims_to_operand_dims: ArrayLikeSpec,
    ) -> None:
        """Initialize ``ScatterDimensionNumbers``.

        Args:
            update_window_dims: Dimensions in the updates tensor that are window dimensions.
            inserted_window_dims: Dimensions in the operand tensor that are inserted into windows.
            scatter_dims_to_operand_dims: Mapping from scatter index dimensions to operand dimensions.
        """
        self.update_window_dims = update_window_dims
        self.inserted_window_dims = inserted_window_dims
        self.scatter_dims_to_operand_dims = scatter_dims_to_operand_dims


__all__ = [
    "ConvDimensionNumbers",
    "ConvGeneralDilatedDimensionNumbers",
    "DotDimensionNumbers",
    "GatherDimensionNumbers",
    "GatherScatterMode",
    "Precision",
    "PrecisionLike",
    "RandomAlgorithm",
    "RoundingMethod",
    "ScatterDimensionNumbers",
]
