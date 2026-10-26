"""Tests for zero_jax.interpreters.traceback_util.util.jaxlib_utils."""

from __future__ import annotations

import pytest

import zero_jax.interpreters.traceback_util.util.jaxlib_utils as mod


def test_module_exists() -> None:
    """Test module."""
    op = mod.CallOp("test_op")
    assert op.name == "test_op"
    assert callable(mod.safe_map)


def test_safe_map() -> None:
    """Test safe_map."""
    assert mod.safe_map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    with pytest.raises(ValueError):
        mod.safe_map(lambda x, y: x + y, [1, 2], [1])
    assert mod.safe_map(lambda x: x) == []


def test_safe_zip() -> None:
    """Test safe_zip."""
    assert mod.safe_zip([1, 2], [3, 4]) == [(1, 3), (2, 4)]
    with pytest.raises(ValueError):
        mod.safe_zip([1, 2], [3])
    assert mod.safe_zip() == []
