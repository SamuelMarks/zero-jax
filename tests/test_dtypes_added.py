import pytest
import zero_jax.dtypes as mod


def test_float0():
    assert hasattr(mod, "float0")
