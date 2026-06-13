import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler


@pytest.fixture(autouse=True)
def reset_tracing_state():
    from ml_switcheroo_compiler.tracing import _tracer

    _tracer.is_tracing = False
    _tracer.active_graph = None
    yield
    _tracer.is_tracing = False
    _tracer.active_graph = None


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    with ml_switcheroo_compiler.EagerMode():
        yield


try:
    import jax
    import jax.numpy as jnp_ref

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import zero_jax.numpy as jnp_zero
import numpy as np


@pytest.fixture
def check_allclose():
    if not HAS_JAX:
        pytest.skip("Official JAX is not installed. Skipping reference tests.")

    def _check(zero_val, ref_val, rtol=1e-5, atol=1e-5):
        if isinstance(ref_val, (jax.Array, np.ndarray, float, int, bool)):
            np.testing.assert_allclose(zero_val, ref_val, rtol=rtol, atol=atol)
        elif isinstance(ref_val, tuple):
            assert len(zero_val) == len(ref_val)
            for z, r in zip(zero_val, ref_val):
                _check(z, r, rtol, atol)
        elif isinstance(ref_val, list):
            assert len(zero_val) == len(ref_val)
            for z, r in zip(zero_val, ref_val):
                _check(z, r, rtol, atol)
        elif isinstance(ref_val, dict):
            assert set(zero_val.keys()) == set(ref_val.keys())
            for k in ref_val:
                _check(zero_val[k], ref_val[k], rtol, atol)
        else:
            assert zero_val == ref_val

    return _check
