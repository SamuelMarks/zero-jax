"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.cluster.vq as mod


def test_vq() -> None:
    """Test vq."""
    with patch("ml_switcheroo_compiler.ops.vq") as mock_op:
        mod.vq()
        mock_op.assert_called_once_with()
