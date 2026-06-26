from ml_switcheroo_compiler.core.tensor import TensorConfig

"""Tests for zero_jax.random."""

import numpy as np
from zero_jax.random import split, fold_in
from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor


def test_random_split_eager():
    key = np.array([0, 0])
    keys = split(key, 2)
    assert keys.shape == (2, 2)
    split(key, 1)


def test_random_fold_in_eager():
    key = np.array([0, 0])
    new_key = fold_in(key, 5)
    assert new_key.shape == (2,)


def test_random_tracing():
    _tracer.start_tracing()
    key = ProxyTensor(id="k", shape=(2,))
    keys = split(key, 2)
    assert keys.shape == (2, 2)

    new_key = fold_in(key, 5)
    assert new_key.shape == (2,)

    fold_in(key, ProxyTensor(id="d", shape=()))
    _tracer.stop_tracing()


def test_random_fallback():
    assert split(None) is None
    assert fold_in(None, 5) is None


def test_truncated_normal():
    from zero_jax import random
    import numpy as np

    key = random.PRNGKey(0)
    x = random.truncated_normal(key, -1.0, 1.0, shape=(2, 2))
    assert x.shape == (2, 2)
