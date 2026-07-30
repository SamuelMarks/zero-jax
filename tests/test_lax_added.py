import pytest

from zero_jax import lax


def test_array_device() -> None:
    """Test Array and Device."""
    assert lax.Array is not None
    assert lax.Device is not None
