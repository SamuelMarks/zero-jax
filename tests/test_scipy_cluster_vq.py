"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.cluster.vq as mod


def test_vq() -> None:
    """Test vq."""
    with pytest.raises(NotImplementedError):
        mod.vq()
