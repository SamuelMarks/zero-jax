"""Tests for zero_jax.stages."""

from __future__ import annotations

import zero_jax.stages as mod


def test_class_ArgInfo() -> None:
    """Test class ArgInfo."""
    info = mod.ArgInfo(aval="float32[2]", donated=True, shape=(2,), dtype="float32")
    assert info.donated is True
    assert info.shape == (2,)
    assert info.dtype == "float32"


def test_class_OutInfo() -> None:
    """Test class OutInfo."""
    out = mod.OutInfo(shape=(3,), dtype="int32")
    assert out.shape == (3,)
    assert out.dtype == "int32"


def test_class_CompilerOptions() -> None:
    """Test class CompilerOptions."""
    opts = mod.CompilerOptions(xla_embed_constants=True)
    assert opts["xla_embed_constants"] is True


def test_class_Wrapped_Traced_Lowered_Compiled() -> None:
    """Test full Wrapped -> Traced -> Lowered -> Compiled pipeline."""
    wrapped = mod.Wrapped(lambda x: x)
    assert wrapped(5) == 5

    empty_wrapped = mod.Wrapped()
    assert empty_wrapped(5) is None

    traced = wrapped.trace(5)
    assert isinstance(traced, mod.Traced)

    lowered = wrapped.lower(5)
    assert isinstance(lowered, mod.Lowered)
    assert "module" in lowered.as_text()
    assert lowered.cost_analysis() == {"flops": 0.0}

    compiled = lowered.compile()
    assert isinstance(compiled, mod.Compiled)
    assert "module" in compiled.as_text()
    assert "flops" in compiled.cost_analysis()
    assert "temp_size_in_bytes" in compiled.memory_analysis()
    assert compiled(7) == 7

    empty_compiled = mod.Compiled()
    assert empty_compiled(7) is None
