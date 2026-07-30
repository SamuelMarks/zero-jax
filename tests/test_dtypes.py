"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.dtypes as mod


def test_bfloat16() -> None:
    """Test bfloat16."""
    obj = mod.bfloat16(dummy=1)
    assert obj is not None


def test_canonicalize_dtype() -> None:
    """Test canonicalize_dtype."""
    with patch(
        "zero_jax._compiler_proxy_ops.canonicalize_dtype", create=True
    ) as mock_op:
        mod.canonicalize_dtype()
        mock_op.assert_called_once_with()


def test_extended() -> None:
    """Test extended."""
    obj = mod.extended(dummy=1)
    assert obj is not None


def test_finfo() -> None:
    """Test finfo."""
    obj = mod.finfo(dummy=1)
    assert obj is not None


def test_iinfo() -> None:
    """Test iinfo."""
    obj = mod.iinfo(dummy=1)
    assert obj is not None


def test_issubdtype() -> None:
    """Test issubdtype."""
    with patch("zero_jax._compiler_proxy_ops.issubdtype", create=True) as mock_op:
        mod.issubdtype()
        mock_op.assert_called_once_with()


def test_prng_key() -> None:
    """Test prng_key."""
    obj = mod.prng_key(dummy=1)
    assert obj is not None


def test_result_type() -> None:
    """Test result_type."""
    with patch("zero_jax._compiler_proxy_ops.result_type", create=True) as mock_op:
        mod.result_type()
        mock_op.assert_called_once_with()


def test_scalar_type_of() -> None:
    """Test scalar_type_of."""
    with patch("zero_jax._compiler_proxy_ops.scalar_type_of", create=True) as mock_op:
        mod.scalar_type_of()
        mock_op.assert_called_once_with()
