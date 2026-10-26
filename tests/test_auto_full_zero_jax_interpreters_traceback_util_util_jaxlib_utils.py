"""Tests for zero_jax.interpreters.traceback_util.util.jaxlib_utils."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.jaxlib_utils as mod


def test_class_CallOp() -> None:
    """Test class CallOp."""
    obj = mod.CallOp()
    assert isinstance(obj, mod.CallOp)


def test_safe_map() -> None:
    """Test safe_map."""
    res = mod.safe_map(lambda x: x + 1, [1, 2, 3])
    assert res == [2, 3, 4]


def test_safe_zip() -> None:
    """Test safe_zip."""
    res = mod.safe_zip([1, 2], ["a", "b"])
    assert res == [(1, "a"), (2, "b")]
