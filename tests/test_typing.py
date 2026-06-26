"""Tests for zero_jax module."""

import pytest
import zero_jax.typing as mod


def test_ArrayLike() -> None:
    """Test ArrayLike."""
    with pytest.raises(NotImplementedError):
        mod.ArrayLike()


def test_DTypeLike() -> None:
    """Test DTypeLike."""
    with pytest.raises(NotImplementedError):
        mod.DTypeLike()
