"""Tests for tests/test_rrelu.py."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops


def test_rrelu_exists() -> None:
    """Execute ``test_rrelu_exists``."""
    assert hasattr(ops, "rrelu")
