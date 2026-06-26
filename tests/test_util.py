"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.util as mod


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction()
    assert obj is not None


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    with patch("ml_switcheroo_compiler.ops.as_hashable_function") as mock_op:
        mod.as_hashable_function()
        mock_op.assert_called_once_with()


def test_cache() -> None:
    """Test cache."""
    with patch("ml_switcheroo_compiler.ops.cache") as mock_op:
        mod.cache()
        mock_op.assert_called_once_with()


def test_safe_map() -> None:
    """Test safe_map."""
    with patch("ml_switcheroo_compiler.ops.safe_map") as mock_op:
        mod.safe_map()
        mock_op.assert_called_once_with()


def test_safe_zip() -> None:
    """Test safe_zip."""
    with patch("ml_switcheroo_compiler.ops.safe_zip") as mock_op:
        mod.safe_zip()
        mock_op.assert_called_once_with()


def test_split_dict() -> None:
    """Test split_dict."""
    with patch("ml_switcheroo_compiler.ops.split_dict") as mock_op:
        mod.split_dict()
        mock_op.assert_called_once_with()


def test_split_list() -> None:
    """Test split_list."""
    with patch("ml_switcheroo_compiler.ops.split_list") as mock_op:
        mod.split_list()
        mock_op.assert_called_once_with()


def test_split_list_checked() -> None:
    """Test split_list_checked."""
    with patch("ml_switcheroo_compiler.ops.split_list_checked") as mock_op:
        mod.split_list_checked()
        mock_op.assert_called_once_with()


def test_split_merge() -> None:
    """Test split_merge."""
    with patch("ml_switcheroo_compiler.ops.split_merge") as mock_op:
        mod.split_merge()
        mock_op.assert_called_once_with()


def test_subvals() -> None:
    """Test subvals."""
    with patch("ml_switcheroo_compiler.ops.subvals") as mock_op:
        mod.subvals()
        mock_op.assert_called_once_with()


def test_toposort() -> None:
    """Test toposort."""
    with patch("ml_switcheroo_compiler.ops.toposort") as mock_op:
        mod.toposort()
        mock_op.assert_called_once_with()


def test_unzip2() -> None:
    """Test unzip2."""
    with patch("ml_switcheroo_compiler.ops.unzip2") as mock_op:
        mod.unzip2()
        mock_op.assert_called_once_with()


def test_wrap_name() -> None:
    """Test wrap_name."""
    with patch("ml_switcheroo_compiler.ops.wrap_name") as mock_op:
        mod.wrap_name()
        mock_op.assert_called_once_with()


def test_wraps() -> None:
    """Test wraps."""
    with patch("ml_switcheroo_compiler.ops.wraps") as mock_op:
        mod.wraps()
        mock_op.assert_called_once_with()
