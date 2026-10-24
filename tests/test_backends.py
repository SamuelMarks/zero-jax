"""Tests for tests/test_backends.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ml_switcheroo_compiler.core.config import config

import zero_jax
from zero_jax import _compiler_patches, _compiler_proxy_creation
from zero_jax import numpy as jnp
from zero_jax import random as jrandom


def test_devices() -> None:
    """Verify device platform reporting across backends."""
    config.backend = "mlx"
    d = zero_jax.devices()
    assert len(d) == 1
    assert d[0].platform == "mlx"

    config.backend = "dask"
    assert zero_jax.local_devices()[0].platform == "dask"

    config.backend = "cupy"
    assert zero_jax.devices()[0].platform == "cupy"

    config.backend = "numpy"


def test_deterministic_prng_compiler_patches() -> None:
    """Verify deterministic PRNG behavior and key seeding in ``_compiler_patches``."""
    _compiler_patches.register_compiler_patches()
    assert _compiler_patches._seed_from_key(np, np.array([], dtype=np.uint32)) == 0
    k1 = jrandom.PRNGKey(42)
    k2 = jrandom.PRNGKey(42)
    k3 = jrandom.PRNGKey(99)

    n1 = jrandom.normal(k1, shape=(4,))
    n2 = jrandom.normal(k2, shape=(4,))
    n3 = jrandom.normal(k3, shape=(4,))
    np.testing.assert_allclose(np.asarray(n1), np.asarray(n2))
    assert not np.allclose(np.asarray(n1), np.asarray(n3))

    r1 = jrandom.randint(k1, shape=(6,), minval=0, maxval=100)
    r2 = jrandom.randint(k2, shape=(6,), minval=0, maxval=100)
    r3 = jrandom.randint(k3, shape=(6,), minval=0, maxval=100)
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))
    assert not np.array_equal(np.asarray(r1), np.asarray(r3))

    b1 = jrandom.bernoulli(k1, p=0.5, shape=(16,))
    b2 = jrandom.bernoulli(k2, p=0.5, shape=(16,))
    b3 = jrandom.bernoulli(k3, p=0.5, shape=(16,))
    np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))
    assert not np.array_equal(np.asarray(b1), np.asarray(b3))

    # Exercise direct fallback with empty positional args
    assert _compiler_patches._np_random_normal(np, shape=(2,)).shape == (2,)
    assert _compiler_patches._np_random_randint(np, shape=(2,), minval=0, maxval=5).shape == (2,)
    assert _compiler_patches._np_random_bernoulli(np, shape=(2,), p=0.5).shape == (2,)


def test_compiler_proxy_creation_shape_validation() -> None:
    """Verify shape validation in ``_compiler_proxy_creation``."""
    assert _compiler_proxy_creation.zeros(3).shape == (3,)
    assert _compiler_proxy_creation.ones([2, 3]).shape == (2, 3)

    bad_shapes: list[zero_jax._types.ConfigValue] = ["invalid", 3.14]
    for bad_s in bad_shapes:
        with pytest.raises(TypeError, match="Shape must be an int or sequence of ints"):
            _compiler_proxy_creation._validate_shape(bad_s)

    bad_seq_shapes: list[zero_jax._types.ConfigValue] = [["bad"], [1.5]]
    for bad_seq in bad_seq_shapes:
        with pytest.raises(TypeError, match="Shape dimensions must be integers"):
            _compiler_proxy_creation._validate_shape(bad_seq)


def test_from_dlpack() -> None:
    """Execute ``test_from_dlpack``."""
    try:
        import torch
    except ImportError:
        return

    x = torch.tensor([1, 2, 3])

    # zero_jax.dlpack.from_dlpack
    out = zero_jax.dlpack.from_dlpack(x)
    assert np.allclose(np.array(out), np.array([1, 2, 3]))

    # zero_jax.numpy.from_dlpack
    out2 = jnp.from_dlpack(x)
    assert np.allclose(np.array(out2), np.array([1, 2, 3]))
