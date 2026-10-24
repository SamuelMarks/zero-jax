"""Tests for zero_jax.nn.initializers."""

from __future__ import annotations

import numpy as np

import zero_jax.nn.initializers as mod


def test_all_initializers_concrete() -> None:
    """Verify all initializers produce arrays of the expected shape and properties."""
    key = 42
    assert mod.zeros(key, (2, 3)).shape == (2, 3)
    assert mod.ones(key, (2, 3)).shape == (2, 3)
    assert np.allclose(np.asarray(mod.constant(2.5)(key, (2, 2))), 2.5)
    assert mod.uniform(0.05)(key, (2, 3)).shape == (2, 3)
    assert mod.normal(0.05)(key, (2, 3)).shape == (2, 3)
    assert mod.truncated_normal(0.05)(key, (2, 3)).shape == (2, 3)
    assert mod.variance_scaling(1.0, "fan_avg", "normal")(key, (3, 4)).shape == (3, 4)
    assert mod.glorot_normal()(key, (3, 4)).shape == (3, 4)
    assert mod.glorot_uniform()(key, (3, 4)).shape == (3, 4)
    assert mod.he_normal()(key, (3, 4)).shape == (3, 4)
    assert mod.he_uniform()(key, (3, 4)).shape == (3, 4)
    assert mod.kaiming_normal()(key, (3, 4)).shape == (3, 4)
    assert mod.kaiming_uniform()(key, (3, 4)).shape == (3, 4)
    assert mod.lecun_normal()(key, (3, 4)).shape == (3, 4)
    assert mod.lecun_uniform()(key, (3, 4)).shape == (3, 4)
    assert mod.xavier_normal()(key, (3, 4)).shape == (3, 4)
    assert mod.xavier_uniform()(key, (3, 4)).shape == (3, 4)
    assert mod.orthogonal()(key, (3, 3)).shape == (3, 3)
    assert mod.delta_orthogonal()(key, (3, 3, 2, 2)).shape == (3, 3, 2, 2)
