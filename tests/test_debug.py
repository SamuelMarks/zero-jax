"""Tests for zero_jax.debug."""

from __future__ import annotations

import zero_jax.debug as mod
import zero_jax.numpy as jnp


def test_DebugEffect() -> None:
    """Test DebugEffect."""
    obj = mod.DebugEffect(id="dbg1", name="effect", value=1)
    assert obj.id == "dbg1"
    assert obj.name == "effect"
    assert obj.value == 1


def test_breakpoint() -> None:
    """Test breakpoint."""
    mod.breakpoint(backend="cli", filter_frames=True)


def test_callback() -> None:
    """Test callback."""
    seen: list[int] = []
    mod.callback(lambda x: seen.append(x), 7)
    assert seen == [7]


def test_inspect_array_sharding() -> None:
    """Test inspect_array_sharding."""
    seen: list[str] = []
    mod.inspect_array_sharding(1, callback=lambda s: seen.append(str(s)))
    assert seen == ["None"]


def test_print() -> None:
    """Test print."""
    mod.print("val={x}", x=3)


def test_visualize_array_sharding() -> None:
    """Test visualize_array_sharding."""
    out = mod.visualize_array_sharding(jnp.ones((2, 2)))
    assert "Array" in out


def test_visualize_sharding() -> None:
    """Test visualize_sharding."""
    assert mod.visualize_sharding("cpu") == "cpu"
