"""Tests for zero_jax module."""

import pytest

import zero_jax.interpreters.traceback_util.util.jaxlib_utils as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None


def test_safe_map() -> None:
    """Test safe_map."""
    assert mod.safe_map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    try:
        mod.safe_map(lambda x, y: x + y, [1, 2], [1])
        assert False
    except AssertionError:
        pass
    assert mod.safe_map(lambda x: x) == []


def test_safe_zip() -> None:
    """Test safe_zip."""
    assert mod.safe_zip([1, 2], [3, 4]) == [(1, 3), (2, 4)]
    try:
        mod.safe_zip([1, 2], [3])
        assert False
    except AssertionError:
        pass
    assert mod.safe_zip() == []
