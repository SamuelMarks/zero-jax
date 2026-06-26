"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed.xla_bridge as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None
