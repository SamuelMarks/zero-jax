"""Tests for zero_jax.lax.types."""

from __future__ import annotations

import zero_jax.lax.types as mod


def test_class_ConvDimensionNumbers() -> None:
    """Test class ConvDimensionNumbers."""
    cdn = mod.ConvDimensionNumbers((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    assert cdn.lhs_spec == (0, 3, 1, 2)
    assert cdn.rhs_spec == (3, 2, 0, 1)
    assert cdn.out_spec == (0, 3, 1, 2)


def test_class_ConvGeneralDilatedDimensionNumbers() -> None:
    """Test class ConvGeneralDilatedDimensionNumbers."""
    cgdn = mod.ConvGeneralDilatedDimensionNumbers((0, 1, 2), (2, 1, 0), (0, 1, 2))
    assert cgdn.lhs_spec == (0, 1, 2)
    assert cgdn.rhs_spec == (2, 1, 0)
    assert cgdn.out_spec == (0, 1, 2)


def test_class_DotDimensionNumbers() -> None:
    """Test class DotDimensionNumbers."""
    ddn = mod.DotDimensionNumbers((1,), (0,), (0,), (0,))
    assert ddn.lhs_contracting == (1,)
    assert ddn.rhs_contracting == (0,)
    assert ddn.lhs_batch == (0,)
    assert ddn.rhs_batch == (0,)


def test_class_GatherDimensionNumbers() -> None:
    """Test class GatherDimensionNumbers."""
    gdn = mod.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))
    assert gdn.offset_dims == (1,)
    assert gdn.collapsed_slice_dims == (0,)
    assert gdn.start_index_map == (0,)


def test_class_GatherScatterMode() -> None:
    """Test class GatherScatterMode."""
    mode = mod.GatherScatterMode()
    assert mode.value == mod.GatherScatterMode.FILL_OR_DROP
    clip_mode = mod.GatherScatterMode.from_any("clip")
    assert clip_mode.value == "CLIP"
    assert mod.GatherScatterMode.from_any(clip_mode) is clip_mode
    default_mode = mod.GatherScatterMode.from_any(None)
    assert default_mode.value == mod.GatherScatterMode.FILL_OR_DROP


def test_class_Precision_and_PrecisionLike() -> None:
    """Test class Precision and PrecisionLike."""
    assert mod.Precision.DEFAULT == "DEFAULT"
    assert mod.Precision.HIGH == "HIGH"
    assert mod.Precision.HIGHEST == "HIGHEST"
    p = mod.PrecisionLike(mod.Precision.HIGH)
    assert p.value == "HIGH"


def test_class_RandomAlgorithm() -> None:
    """Test class RandomAlgorithm."""
    alg = mod.RandomAlgorithm(mod.RandomAlgorithm.RNG_THREE_FRY)
    assert alg.value == "RNG_THREE_FRY"


def test_class_RoundingMethod() -> None:
    """Test class RoundingMethod."""
    rm = mod.RoundingMethod(mod.RoundingMethod.AWAY_FROM_ZERO)
    assert rm.value == 0
    assert mod.RoundingMethod.TO_NEAREST_EVEN == 1


def test_class_ScatterDimensionNumbers() -> None:
    """Test class ScatterDimensionNumbers."""
    sdn = mod.ScatterDimensionNumbers((1,), (0,), (0,))
    assert sdn.update_window_dims == (1,)
    assert sdn.inserted_window_dims == (0,)
    assert sdn.scatter_dims_to_operand_dims == (0,)
