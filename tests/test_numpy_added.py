import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import zero_jax.numpy as jnp
from zero_jax.numpy import array


def test_constants():
    np.testing.assert_allclose(jnp.e, np.e)
    assert jnp.euler_gamma > 0
    assert jnp.newaxis is None


def test_indexers():
    assert jnp.s_[1:2] == slice(1, 2, None)
    assert jnp.index_exp[1:2] == (slice(1, 2, None),)
    # Check shape to ensure array conversion happened
    assert jnp.r_[1:4].shape == (3,)
    assert jnp.c_[1:4].shape == (3, 1)


def test_poly():
    p1 = jnp.array([1.0, 2.0, 3.0])
    p2 = jnp.array([4.0, 5.0, 6.0])

    # Just check they return arrays without crashing
    assert jnp.polyadd(p1, p2).shape is not None
    assert jnp.polysub(p1, p2).shape is not None
    assert jnp.polymul(p1, p2).shape is not None

    q, r = jnp.polydiv(p1, jnp.array([1.0, 1.0]))
    assert q.shape is not None
    assert r.shape is not None

    assert jnp.polyder(p1).shape is not None
    assert jnp.polyint(p1).shape is not None
    assert jnp.polyval(p1, 2.0).shape is not None

    assert jnp.roots(p1).shape is not None
    assert jnp.poly(jnp.array([1.0, 2.0])).shape is not None

    fit = jnp.polyfit(jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 4.0]), 2)
    assert fit.shape is not None


def test_config():
    with jnp.printoptions(precision=3):
        pass
    jnp.set_printoptions(precision=4)


def test_added_funcs():
    a = jnp.array([1, 2, 3])
    b = jnp.array([2, 3, 4])

    assert len(jnp.nonzero(a)) > 0
    assert jnp.packbits(jnp.array([0, 1])).shape is not None
    assert jnp.unpackbits(np.array([0], dtype=np.uint8)).shape is not None

    assert jnp.partition(a, 1).shape is not None
    assert jnp.percentile(a, 50).shape is not None
    assert jnp.permute_dims(jnp.array([[1, 2]]), (1, 0)).shape == (2, 1)

    assert (
        jnp.piecewise(
            np.array([1, 2, 3]),
            [np.array([True, False, False]), np.array([False, True, True])],
            [1, 2],
        ).shape
        is not None
    )

    jnp.place(a, np.array([True, False, False]), [9])

    assert jnp.promote_types(np.float32, np.float64) == np.float64
    assert jnp.ptp(a).shape is not None

    jnp.put(a, [0], [8])

    assert jnp.quantile(a, 0.5).shape is not None

    assert (
        jnp.ravel_multi_index((jnp.array([0]), jnp.array([1])), (2, 2)).shape
        is not None
    )
    assert jnp.resize(a, (5,)).shape == (5,)
    assert jnp.result_type(a, b) == a.dtype.value

    assert jnp.rollaxis(jnp.array([[1]]), 0).shape is not None
    assert jnp.rot90(np.array([[1, 2], [3, 4]])).shape is not None

    jnp.save("test_save.npy", np.array([1, 2, 3]))
    jnp.savez("test_savez.npz", a=np.array([1, 2, 3]))
    os.remove("test_save.npy")
    os.remove("test_savez.npz")

    assert jnp.setdiff1d(a, b).shape is not None
    assert jnp.setxor1d(a, b).shape is not None
    assert jnp.size(a) == 3

    assert jnp.sort_complex(a).shape is not None
    assert jnp.trace(jnp.array([[1, 2], [3, 4]])).shape is not None
    assert jnp.trapezoid(a).shape is not None
    assert jnp.tri(3).shape == (3, 3)

    assert len(jnp.tril_indices(3)) == 2
    assert len(jnp.tril_indices_from(np.array([[1, 2], [3, 4]]))) == 2

    assert jnp.trim_zeros(jnp.array([0, 1, 0])).shape is not None

    assert len(jnp.triu_indices(3)) == 2
    assert len(jnp.triu_indices_from(np.array([[1, 2], [3, 4]]))) == 2

    assert jnp.uint(5) == 5
    assert jnp.uint4(5) == 5

    assert jnp.union1d(a, b).shape is not None
    assert jnp.unique(a).shape is not None
    assert len(jnp.unique_all(a)) == 4
    assert len(jnp.unique_counts(a)) == 2
    assert len(jnp.unique_inverse(a)) == 2
    assert len(jnp.unique_values(a)) == 3

    assert len(jnp.unravel_index(jnp.array([1]), (2, 2))) == 2
    assert jnp.unwrap(a).shape is not None

    assert jnp.vander(a).shape is not None
    assert jnp.vecdot(a, a).shape is not None

    @jnp.vectorize
    def f(x):
        return x + 1

    assert f(a).shape is not None
