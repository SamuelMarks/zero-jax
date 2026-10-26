"""Tests for newly added numpy wrappers."""

from __future__ import annotations

from pathlib import Path

import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_apply_over_axes() -> None:
    """Execute ``test_apply_over_axes``."""
    a = jnp.ones((2, 3, 4))
    res = jnp.apply_over_axes(lambda x, ax: jnp.sum(x, axis=ax, keepdims=True), a, [0, 2])
    assert res.shape == (1, 3, 1)
    assert_allclose(res, [[[8.0], [8.0], [8.0]]])


def test_argwhere() -> None:
    """Execute ``test_argwhere``."""
    res = jnp.argwhere(jnp.array([0, 2, 0, 4]))
    assert_allclose(res, [[1], [3]])


def test_bartlett() -> None:
    """Execute ``test_bartlett``."""
    res = jnp.bartlett(5)
    assert_allclose(res, [0.0, 0.5, 1.0, 0.5, 0.0])


def test_blackman() -> None:
    """Execute ``test_blackman``."""
    res = jnp.blackman(5)
    assert res.shape == (5,)
    assert_allclose(res[2], 1.0, atol=1e-5)


def test_frombuffer() -> None:
    """Execute ``test_frombuffer``."""
    res = jnp.frombuffer([1.0, 2.0, 3.0], dtype="float32")
    assert_allclose(res, [1.0, 2.0, 3.0])


def test_fromfile(tmp_path: Path) -> None:
    """Execute ``test_fromfile``.

    Args:
        tmp_path: Temporary path fixture.
    """
    p = tmp_path / "arr.txt"
    p.write_text("1.0 2.0", encoding="utf-8")
    res = jnp.fromfile(str(p), sep=" ")
    assert_allclose(res, [1.0, 2.0])


def test_fromiter() -> None:
    """Execute ``test_fromiter``."""
    res = jnp.fromiter([1.0, 2.0, 3.0], dtype="float32")
    assert_allclose(res, [1.0, 2.0, 3.0])


def test_fromstring() -> None:
    """Execute ``test_fromstring``."""
    res = jnp.fromstring("1.0,2.0", sep=",")
    assert_allclose(res, [1.0, 2.0])


def test_geomspace() -> None:
    """Execute ``test_geomspace``."""
    res = jnp.geomspace(1.0, 100.0, 3)
    assert_allclose(res, [1.0, 10.0, 100.0], atol=1e-4)


def test_gradient() -> None:
    """Execute ``test_gradient``."""
    res = jnp.gradient(jnp.array([1.0, 2.0, 4.0]))
    assert_allclose(res, [1.0, 1.5, 2.0])


def test_hamming() -> None:
    """Execute ``test_hamming``."""
    res = jnp.hamming(5)
    assert res.shape == (5,)
    assert_allclose(res[2], 1.0, atol=1e-5)


def test_hanning() -> None:
    """Execute ``test_hanning``."""
    res = jnp.hanning(5)
    assert_allclose(res, [0.0, 0.5, 1.0, 0.5, 0.0], atol=1e-5)


def test_histogram() -> None:
    """Execute ``test_histogram``."""
    h, edges = jnp.histogram(jnp.array([0.25, 1.25, 1.75]), bins=2, range=(0.0, 2.0))
    assert_allclose(h, [1.0, 2.0])
    assert_allclose(edges, [0.0, 1.0, 2.0])


def test_histogram2d() -> None:
    """Execute ``test_histogram2d``."""
    h, xe, ye = jnp.histogram2d(jnp.array([0.2, 0.8]), jnp.array([0.2, 0.8]), bins=2, range=((0.0, 1.0), (0.0, 1.0)))
    assert h.shape == (2, 2)
    assert xe.shape == (3,) and ye.shape == (3,)


def test_histogram_bin_edges() -> None:
    """Execute ``test_histogram_bin_edges``."""
    edges = jnp.histogram_bin_edges(jnp.array([0.0, 2.0]), bins=2)
    assert_allclose(edges, [0.0, 1.0, 2.0])


def test_histogramdd() -> None:
    """Execute ``test_histogramdd``."""
    h, edges = jnp.histogramdd(jnp.array([[0.2, 0.2], [0.8, 0.8]]), bins=2)
    assert h.shape == (2, 2)
    assert len(edges) == 2


def test_kaiser() -> None:
    """Execute ``test_kaiser``."""
    res = jnp.kaiser(5, 0.0)
    assert_allclose(res, [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-5)


def test_linalg_cond() -> None:
    """Execute ``test_linalg_cond``."""
    res = jnp.linalg.cond(jnp.eye(2))
    assert_allclose(res, 1.0, atol=1e-4)


def test_eigvals() -> None:
    """Execute ``test_eigvals``."""
    res = jnp.linalg.eigvals(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    assert_allclose(jnp.sort(jnp.real(res)), [2.0, 3.0], atol=1e-4)
