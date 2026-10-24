"""Tests for newly added numpy wrappers."""

from __future__ import annotations

from unittest.mock import patch

import zero_jax.numpy as jnp


def test_apply_over_axes() -> None:
    """Execute ``test_apply_over_axes``."""
    with patch("zero_jax._compiler_proxy_ops.apply_over_axes", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.apply_over_axes(lambda x, axis: x, [1.0], [0]) is not None


def test_argpartition() -> None:
    """Execute ``test_argpartition``."""
    with patch("zero_jax._compiler_proxy_ops.argpartition", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.argpartition([1.0], 0) is not None


def test_argwhere() -> None:
    """Execute ``test_argwhere``."""
    with patch("zero_jax._compiler_proxy_ops.argwhere", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.argwhere([1.0]) is not None


def test_fromfile() -> None:
    """Execute ``test_fromfile``."""
    with patch("zero_jax._compiler_proxy_ops.fromfile", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.fromfile("test.txt") is not None


def test_fromfunction() -> None:
    """Execute ``test_fromfunction``."""
    with patch("zero_jax._compiler_proxy_ops.fromfunction", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.fromfunction(lambda x: x, (2,)) is not None


def test_fromiter() -> None:
    """Execute ``test_fromiter``."""
    with patch("zero_jax._compiler_proxy_ops.fromiter", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.fromiter([1.0], float) is not None


def test_frompyfunc() -> None:
    """Execute ``test_frompyfunc``."""
    with patch("zero_jax._compiler_proxy_ops.frompyfunc", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.frompyfunc(lambda x: x, 1, 1) is not None


def test_fromstring() -> None:
    """Execute ``test_fromstring``."""
    with patch("zero_jax._compiler_proxy_ops.fromstring", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.fromstring("1 2 3", sep=" ") is not None


def test_geomspace() -> None:
    """Execute ``test_geomspace``."""
    with patch("zero_jax._compiler_proxy_ops.geomspace", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.geomspace(1.0, 10.0) is not None


def test_get_printoptions() -> None:
    """Execute ``test_get_printoptions``."""
    with patch("zero_jax._compiler_proxy_ops.get_printoptions", create=True) as mock_op:
        mock_op.return_value = {}
        assert jnp.get_printoptions() is not None


def test_gradient() -> None:
    """Execute ``test_gradient``."""
    with patch("zero_jax._compiler_proxy_ops.gradient", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.gradient([1.0]) is not None


def test_hamming() -> None:
    """Execute ``test_hamming``."""
    with patch("zero_jax._compiler_proxy_ops.hamming", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.hamming(10) is not None


def test_hanning() -> None:
    """Execute ``test_hanning``."""
    with patch("zero_jax._compiler_proxy_ops.hanning", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.hanning(10) is not None


def test_histogram() -> None:
    """Execute ``test_histogram``."""
    with patch("zero_jax._compiler_proxy_ops.histogram", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert jnp.histogram([1.0]) is not None


def test_histogram2d() -> None:
    """Execute ``test_histogram2d``."""
    with patch("zero_jax._compiler_proxy_ops.histogram2d", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0, 1.0)
        assert jnp.histogram2d([1.0], [1.0]) is not None


def test_histogramdd() -> None:
    """Execute ``test_histogramdd``."""
    with patch("zero_jax._compiler_proxy_ops.histogramdd", create=True) as mock_op:
        mock_op.return_value = (1.0, 1.0)
        assert jnp.histogramdd([[1.0]]) is not None


def test_histogram_bin_edges() -> None:
    """Execute ``test_histogram_bin_edges``."""
    with patch("zero_jax._compiler_proxy_ops.histogram_bin_edges", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.histogram_bin_edges([1.0]) is not None


def test_i0() -> None:
    """Execute ``test_i0``."""
    with patch("zero_jax._compiler_proxy_ops.i0", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.i0([1.0]) is not None


def test_eigvals() -> None:
    """Execute ``test_eigvals``."""
    with patch("zero_jax._compiler_proxy_ops.eigvals", create=True) as mock_op:
        mock_op.return_value = 1.0
        assert jnp.linalg.eigvals([1.0]) is not None
