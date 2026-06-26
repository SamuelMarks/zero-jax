"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla as mod


def test_Backend() -> None:
    """Test Backend."""
    obj = mod.Backend()
    assert obj is not None


def test_abstractify() -> None:
    """Test abstractify."""
    with pytest.raises(NotImplementedError):
        mod.abstractify()


def test_apply_primitive() -> None:
    """Test apply_primitive."""
    with pytest.raises(NotImplementedError):
        mod.apply_primitive()


def test_canonicalize_dtype() -> None:
    """Test canonicalize_dtype."""
    with pytest.raises(NotImplementedError):
        mod.canonicalize_dtype()
