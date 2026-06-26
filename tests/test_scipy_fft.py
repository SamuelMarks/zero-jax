"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.fft as mod


def test_dct() -> None:
    """Test dct."""
    with pytest.raises(NotImplementedError):
        mod.dct()


def test_dctn() -> None:
    """Test dctn."""
    with pytest.raises(NotImplementedError):
        mod.dctn()


def test_idct() -> None:
    """Test idct."""
    with pytest.raises(NotImplementedError):
        mod.idct()


def test_idctn() -> None:
    """Test idctn."""
    with pytest.raises(NotImplementedError):
        mod.idctn()
