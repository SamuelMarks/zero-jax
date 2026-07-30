import pytest

import zero_jax.dlpack as mod


def test_supported_dtypes():
    assert hasattr(mod, "SUPPORTED_DTYPES")
