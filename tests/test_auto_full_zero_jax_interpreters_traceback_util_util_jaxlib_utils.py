"""Tests for zero_jax.interpreters.traceback_util.util.jaxlib_utils."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.jaxlib_utils as mod


def test_class_CallOp() -> None:
    """Test class CallOp."""
    try:
        mod.CallOp()
    except Exception:
        pass


def test_safe_map() -> None:
    """Test safe_map."""
    try:
        mod.safe_map(1.0)
    except Exception:
        pass


def test_safe_zip() -> None:
    """Test safe_zip."""
    try:
        mod.safe_zip()
    except Exception:
        pass
