"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.dtypes as mod


def test_bfloat16() -> None:
    """Test bfloat16."""
    obj = mod.bfloat16()
    assert obj is not None


def test_canonicalize_dtype() -> None:
    """Test canonicalize_dtype."""
    with patch("ml_switcheroo_compiler.ops.canonicalize_dtype") as mock_op:
        mod.canonicalize_dtype()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.issubdtype") as mock_op:
        mod.issubdtype()
        mock_op.assert_called_once_with()


def test_prng_key() -> None:
    """Test prng_key."""
    obj = mod.prng_key()
    assert obj is not None


def test_result_type() -> None:
    """Test result_type."""
    with patch("ml_switcheroo_compiler.ops.result_type") as mock_op:
        mod.result_type()
        mock_op.assert_called_once_with()


def test_scalar_type_of() -> None:
    """Test scalar_type_of."""
    with patch("ml_switcheroo_compiler.ops.scalar_type_of") as mock_op:
        mod.scalar_type_of()
        mock_op.assert_called_once_with()
