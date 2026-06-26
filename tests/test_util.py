"""Tests for zero_jax module."""

import pytest
import zero_jax.util as mod


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction()
    assert obj is not None


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    with pytest.raises(NotImplementedError):
        mod.as_hashable_function()


def test_cache() -> None:
    """Test cache."""
    with pytest.raises(NotImplementedError):
        mod.cache()


def test_safe_map() -> None:
    """Test safe_map."""
    with pytest.raises(NotImplementedError):
        mod.safe_map()


def test_safe_zip() -> None:
    """Test safe_zip."""
    with pytest.raises(NotImplementedError):
        mod.safe_zip()


def test_split_dict() -> None:
    """Test split_dict."""
    with pytest.raises(NotImplementedError):
        mod.split_dict()


def test_split_list() -> None:
    """Test split_list."""
    with pytest.raises(NotImplementedError):
        mod.split_list()


def test_split_list_checked() -> None:
    """Test split_list_checked."""
    with pytest.raises(NotImplementedError):
        mod.split_list_checked()


def test_split_merge() -> None:
    """Test split_merge."""
    with pytest.raises(NotImplementedError):
        mod.split_merge()


def test_subvals() -> None:
    """Test subvals."""
    with pytest.raises(NotImplementedError):
        mod.subvals()


def test_toposort() -> None:
    """Test toposort."""
    with pytest.raises(NotImplementedError):
        mod.toposort()


def test_unzip2() -> None:
    """Test unzip2."""
    with pytest.raises(NotImplementedError):
        mod.unzip2()


def test_wrap_name() -> None:
    """Test wrap_name."""
    with pytest.raises(NotImplementedError):
        mod.wrap_name()


def test_wraps() -> None:
    """Test wraps."""
    with pytest.raises(NotImplementedError):
        mod.wraps()
