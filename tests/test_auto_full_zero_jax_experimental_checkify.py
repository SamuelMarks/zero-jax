"""Tests for zero_jax.experimental.checkify."""

from __future__ import annotations

import pytest

import zero_jax.experimental.checkify as mod
from zero_jax._types import ArrayLikeSpec


def test_class_ErrorCategory() -> None:
    """Test class ErrorCategory."""
    cat = mod.ErrorCategory("custom", id=1, value="v")
    assert cat.name == "custom"
    assert cat.id == 1
    assert cat.value == "v"


def test_check_and_checkify() -> None:
    """Test check and checkify integration."""

    def fn(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Checked function."""
        mod.check(x != 0, "zero division")
        return x

    checked = mod.checkify(fn)
    err1, out1 = checked(2)
    assert err1 is None
    assert out1 == 2

    err2, out2 = checked(0)
    assert err2 is not None
    assert out2 == 0
    with pytest.raises(mod.JaxRuntimeError, match="zero division"):
        err2.throw()
