"""Tests for zero_jax module."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.typing as mod


def test_ArrayLike() -> None:
    """Test ArrayLike."""
    with patch("zero_jax._compiler_proxy_ops.ArrayLike", create=True) as mock_op:
        mod.ArrayLike()
        mock_op.assert_called_once_with()


def test_DTypeLike() -> None:
    """Test DTypeLike."""
    with patch("zero_jax._compiler_proxy_ops.DTypeLike", create=True) as mock_op:
        mod.DTypeLike()
        mock_op.assert_called_once_with()
