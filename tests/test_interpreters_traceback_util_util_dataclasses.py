"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.traceback_util.util.dataclasses as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None
