"""Tests for zero_jax.version.datetime module."""

from __future__ import annotations

import zero_jax.version.datetime as mod


def test_module_exists() -> None:
    """Verify zero_jax.version.datetime exports standard date and datetime classes."""
    d = mod.date(2026, 10, 26)
    assert d.year == 2026
    assert mod.timezone.utc is not None
