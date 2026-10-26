"""Tests for zero_jax.experimental.checkify."""

from __future__ import annotations

import pytest

import zero_jax.experimental.checkify as mod
from zero_jax._types import ArrayLikeSpec


def test_class_ErrorCategory() -> None:
    """Test class ErrorCategory."""
    cat = mod.ErrorCategory("custom")
    assert cat.name == "custom"
    assert cat.category == "custom"


def test_check_and_checkify() -> None:
    """Test check and checkify integration."""

    def fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Checked function.

        Args:
            x: Parameter ``x``.
        """
        mod.check(x != 0, "zero division")
        return x

    checked = mod.checkify(fn)
    err1, out1 = checked(2)
    assert err1 is None
    assert out1 == 2

    err2, out2 = checked(0)
    assert isinstance(err2, mod.Error)
    assert err2.get() == "zero division"
    assert out2 == 0
    with pytest.raises(mod.JaxRuntimeError, match="zero division"):
        err2.throw()
