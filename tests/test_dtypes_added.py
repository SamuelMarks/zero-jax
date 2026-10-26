"""Tests for tests/test_dtypes_added.py."""

from __future__ import annotations

import zero_jax.dtypes as mod


def test_float0() -> None:
    """Verify float0 returns the symbolic tangent dtype identifier."""
    from zero_jax._config_internal import enable_x64

    assert mod.float0() == "float0"
    assert mod.canonicalize_dtype("float64") == "float32"
    with enable_x64(True):
        assert mod.canonicalize_dtype("float64") == "float64"
        assert mod.canonicalize_dtype("double") == "float64"
        assert mod.canonicalize_dtype("long") == "int64"
