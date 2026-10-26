"""Tests for tests/test_numpy_added3.py."""

from __future__ import annotations

import zero_jax.numpy as jnp
from tests.conftest import assert_allclose
from zero_jax.lax import missing_funcs


def test_section_4_coverage() -> None:
    """Verify real execution of numpy indexing, grids, triangular, set, and polynomial helpers."""
    iota = missing_funcs.broadcasted_iota(3, (2, 3), 1)
    assert iota.shape == (2, 3)

    packed = jnp.packbits(jnp.array([1, 0, 1, 0, 0, 0, 0, 0], dtype="uint8"))
    assert packed.tolist() == [160]
    unpacked = jnp.unpackbits(jnp.array([160], dtype="uint8"))
    assert unpacked.tolist() == [1, 0, 1, 0, 0, 0, 0, 0]

    x = jnp.array([1.0, 2.0, 3.0])
    assert_allclose(jnp.percentile(x, 50), 2.0)
    assert_allclose(jnp.quantile(x, 0.5), 2.0)
    assert_allclose(jnp.piecewise(x, [x < 2.0, x >= 2.0], [0.0, 1.0]), [0.0, 1.0, 1.0])
    assert_allclose(jnp.piecewise(x, x < 2.0, [lambda v: v * 0.0, lambda v: v + 10.0]), [0.0, 12.0, 13.0])
    assert "float" in str(jnp.promote_types("int32", "float32"))
    assert_allclose(jnp.trapezoid(x), 4.0)
    assert_allclose(jnp.tri(2), [[1.0, 0.0], [1.0, 1.0]])
    assert len(jnp.tril_indices(3)) == 2
    assert len(jnp.tril_indices_from(jnp.eye(2))) == 2
    assert len(jnp.triu_indices(3)) == 2
    assert len(jnp.triu_indices_from(jnp.eye(2))) == 2
    assert_allclose(jnp.trim_zeros(jnp.array([0, 1, 2, 0])), [1, 2])
    assert_allclose(jnp.union1d(jnp.array([1, 2]), jnp.array([2, 3])), [1, 2, 3])
    assert_allclose(jnp.unique(jnp.array([2, 1, 2])), [1, 2])
    assert_allclose(jnp.unwrap(jnp.array([0.0, 0.5])), [0.0, 0.5])
    assert_allclose(jnp.vander(jnp.array([1.0, 2.0]), 2), [[1.0, 1.0], [2.0, 1.0]])
    assert len(jnp.nonzero(jnp.array([0, 1, 2]))) == 1

    assert_allclose(jnp.mgrid[0:3], [0, 1, 2])
    assert jnp.mgrid[0:2, 0:3].shape == (2, 2, 3)
    assert_allclose(jnp.ogrid[0:3], [0, 1, 2])
    assert len(jnp.ogrid[0:2, 0:3]) == 2
    assert_allclose(jnp.r_[0:3], [0, 1, 2])
    assert_allclose(jnp.r_[0:2, jnp.array([5, 6])], [0, 1, 5, 6])
    assert_allclose(jnp.r_[jnp.array([1, 2])], [1, 2])
    assert_allclose(jnp.c_[0:2], [[0], [1]])
    assert_allclose(jnp.c_[jnp.array([1, 2]), jnp.array([[3], [4]])], [[1, 3], [2, 4]])
    assert_allclose(jnp.c_[jnp.array([1, 2])], [[1], [2]])
    assert_allclose(jnp.c_[jnp.array([[1, 2]])], [[1, 2]])

    assert_allclose(jnp.poly(jnp.array([1.0, 2.0])), [1.0, -3.0, 2.0])
    assert_allclose(jnp.polyadd(jnp.array([1.0]), jnp.array([2.0])), [3.0])
    assert_allclose(jnp.polyder(jnp.array([3.0, 2.0, 1.0])), [6.0, 2.0])
    q_p, r_p = jnp.polydiv(jnp.array([1.0, 0.0, -1.0]), jnp.array([1.0, -1.0]))
    assert_allclose(q_p, [1.0, 1.0])
    assert_allclose(r_p, [0.0, 0.0, 0.0])
    assert jnp.polyfit(jnp.array([0.0, 1.0]), jnp.array([1.0, 3.0]), 1).shape == (2,)
    assert_allclose(jnp.polyint(jnp.array([2.0, 1.0])), [1.0, 1.0, 0.0])
    assert_allclose(jnp.polymul(jnp.array([1.0, 1.0]), jnp.array([1.0, -1.0])), [1.0, 0.0, -1.0])
    assert_allclose(jnp.polysub(jnp.array([3.0]), jnp.array([1.0])), [2.0])
    assert_allclose(jnp.polyval(jnp.array([2.0, 1.0]), jnp.array(3.0)), 7.0)
    assert jnp.roots(jnp.array([1.0, -3.0, 2.0])).shape == (2,)
