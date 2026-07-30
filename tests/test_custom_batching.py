"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.custom_batching as mod


def test_custom_vmap() -> None:
    """Test custom_vmap."""
    obj = mod.custom_vmap()
    assert obj is not None


def test_sequential_vmap() -> None:
    """Test sequential_vmap."""
    with patch("zero_jax._compiler_proxy_ops.sequential_vmap", create=True) as mock_op:
        mod.sequential_vmap()
        mock_op.assert_called_once_with()
