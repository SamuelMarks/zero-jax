"""Tests for zero_jax module."""

import pytest
import zero_jax.dtypes as mod


def test_bfloat16() -> None:
    """Test bfloat16."""
    obj = mod.bfloat16()
    assert obj is not None


def test_canonicalize_dtype() -> None:
    """Test canonicalize_dtype."""
    with pytest.raises(NotImplementedError):
        mod.canonicalize_dtype()


def test_extended() -> None:
    """Test extended."""
    obj = mod.extended()
    assert obj is not None


def test_finfo() -> None:
    """Test finfo."""
    obj = mod.finfo()
    assert obj is not None


def test_iinfo() -> None:
    """Test iinfo."""
    obj = mod.iinfo()
    assert obj is not None


def test_issubdtype() -> None:
    """Test issubdtype."""
    with pytest.raises(NotImplementedError):
        mod.issubdtype()


def test_prng_key() -> None:
    """Test prng_key."""
    obj = mod.prng_key()
    assert obj is not None


def test_result_type() -> None:
    """Test result_type."""
    with pytest.raises(NotImplementedError):
        mod.result_type()


def test_scalar_type_of() -> None:
    """Test scalar_type_of."""
    with pytest.raises(NotImplementedError):
        mod.scalar_type_of()
