"""Tests for tests/test_dlpack_added.py."""

from __future__ import annotations

import zero_jax.dlpack as mod


def test_supported_dtypes() -> None:
    """Execute ``test_supported_dtypes``."""
    assert mod.SUPPORTED_DTYPES is not None
    assert "SUPPORTED_DTYPES" in dir(mod)
