"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.util as mod


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction()
    assert obj is not None


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    with patch(
        "zero_jax._compiler_proxy_ops.as_hashable_function", create=True
    ) as mock_op:
        mod.as_hashable_function()
        mock_op.assert_called_once_with()


def test_cache() -> None:
    """Test cache."""
    with patch("zero_jax._compiler_proxy_ops.cache", create=True) as mock_op:
        mod.cache()
        mock_op.assert_called_once_with()


def test_safe_map() -> None:
    """Test safe_map."""
    with patch("zero_jax._compiler_proxy_ops.safe_map", create=True) as mock_op:
        mod.safe_map()
        mock_op.assert_called_once_with()


def test_safe_zip() -> None:
    """Test safe_zip."""
    with patch("zero_jax._compiler_proxy_ops.safe_zip", create=True) as mock_op:
        mod.safe_zip()
        mock_op.assert_called_once_with()


def test_split_dict() -> None:
    """Test split_dict."""
    with patch("zero_jax._compiler_proxy_ops.split_dict", create=True) as mock_op:
        mod.split_dict()
        mock_op.assert_called_once_with()


def test_split_list() -> None:
    """Test split_list."""
    with patch("zero_jax._compiler_proxy_ops.split_list", create=True) as mock_op:
        mod.split_list()
        mock_op.assert_called_once_with()


def test_split_list_checked() -> None:
    """Test split_list_checked."""
    with patch(
        "zero_jax._compiler_proxy_ops.split_list_checked", create=True
    ) as mock_op:
        mod.split_list_checked()
        mock_op.assert_called_once_with()


def test_split_merge() -> None:
    """Test split_merge."""
    with patch("zero_jax._compiler_proxy_ops.split_merge", create=True) as mock_op:
        mod.split_merge()
        mock_op.assert_called_once_with()


def test_subvals() -> None:
    """Test subvals."""
    with patch("zero_jax._compiler_proxy_ops.subvals", create=True) as mock_op:
        mod.subvals()
        mock_op.assert_called_once_with()


def test_toposort() -> None:
    """Test toposort."""
    with patch("zero_jax._compiler_proxy_ops.toposort", create=True) as mock_op:
        mod.toposort()
        mock_op.assert_called_once_with()


def test_unzip2() -> None:
    """Test unzip2."""
    with patch("zero_jax._compiler_proxy_ops.unzip2", create=True) as mock_op:
        mod.unzip2()
        mock_op.assert_called_once_with()


def test_wrap_name() -> None:
    """Test wrap_name."""
    with patch("zero_jax._compiler_proxy_ops.wrap_name", create=True) as mock_op:
        mod.wrap_name()
        mock_op.assert_called_once_with()


def test_wraps() -> None:
    """Test wraps."""
    with patch("zero_jax._compiler_proxy_ops.wraps", create=True) as mock_op:
        mod.wraps()
        mock_op.assert_called_once_with()
