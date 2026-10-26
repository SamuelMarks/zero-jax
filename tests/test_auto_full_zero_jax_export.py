"""Tests for zero_jax.export."""

from __future__ import annotations

import zero_jax.export as mod
import zero_jax.numpy as jnp


def test_class_DisabledSafetyCheck() -> None:
    """Test class DisabledSafetyCheck."""
    chk = mod.DisabledSafetyCheck("custom", "op")
    assert chk.kind == "custom"
    assert isinstance(mod.DisabledSafetyCheck.apply(), mod.DisabledSafetyCheck)


def test_class_Exported() -> None:
    """Test class Exported."""
    exp = mod.Exported(fun_name="add_one", fun=lambda x: x + 1)
    assert exp.call(2) == 3
    assert exp.serialize() == bytearray(b"add_one")
    empty_exp = mod.Exported()
    assert empty_exp.fun_name == "exported"
    assert empty_exp.call(2) is None
    assert isinstance(mod.Exported.apply(), mod.Exported)


def test_class_SymbolicScope() -> None:
    """Test class SymbolicScope."""
    scope = mod.SymbolicScope(("b >= 1",))
    assert scope.constraints == ("b >= 1",)
    assert isinstance(mod.SymbolicScope.apply(), mod.SymbolicScope)


def test_default_export_platform() -> None:
    """Test default_export_platform."""
    assert mod.default_export_platform() == "cpu"


def test_deserialize() -> None:
    """Test deserialize."""
    exp = mod.deserialize(b"my_fn")
    assert isinstance(exp, mod.Exported)
    assert exp.fun_name == "my_fn"
    assert isinstance(mod.deserialize(), mod.Exported)


def test_export() -> None:
    """Test export."""
    exported_fn = mod.export(lambda x: x * 2)(jnp.array(3.0))
    assert isinstance(exported_fn, mod.Exported)
    assert exported_fn.call(4) == 8
    assert isinstance(mod.export(), mod.Exported)


def test_is_symbolic_dim() -> None:
    """Test is_symbolic_dim."""
    assert mod.is_symbolic_dim("b") is True
    assert mod.is_symbolic_dim(4) is False
    assert mod.is_symbolic_dim() is False


def test_maximum_supported_calling_convention_version() -> None:
    """Test maximum_supported_calling_convention_version."""
    assert mod.maximum_supported_calling_convention_version() == 9


def test_minimum_supported_calling_convention_version() -> None:
    """Test minimum_supported_calling_convention_version."""
    assert mod.minimum_supported_calling_convention_version() == 9


def test_symbolic_args_specs() -> None:
    """Test symbolic_args_specs."""
    assert mod.symbolic_args_specs((1, 2)) == (1, 2)
    assert mod.symbolic_args_specs(()) == ()


def test_symbolic_shape() -> None:
    """Test symbolic_shape."""
    assert mod.symbolic_shape("b, 4") == ("b", 4)
    assert mod.symbolic_shape() == ()
