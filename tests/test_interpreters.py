"""Tests for zero_jax.interpreters module."""

from __future__ import annotations

import zero_jax.interpreters as mod


def test_module_exists() -> None:
    """Verify zero_jax.interpreters exports core interpreter submodules."""
    assert mod.ad.__name__.endswith("ad")
    assert mod.batching.__name__.endswith("batching")
    assert mod.partial_eval.__name__.endswith("partial_eval")
    assert mod.pxla.__name__.endswith("pxla")
