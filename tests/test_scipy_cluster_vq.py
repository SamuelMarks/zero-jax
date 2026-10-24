"""Tests for zero_jax module."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.scipy.cluster.vq as mod


def test_vq() -> None:
    """Test vq."""
    with patch("zero_jax._compiler_proxy_ops.vq", create=True) as mock_op:
        mod.vq()
        mock_op.assert_called_once_with()
