import pytest

import zero_jax.numpy as jnp


def test_complex_warning() -> None:
    """Test ComplexWarning."""
    assert jnp.ComplexWarning is not None
