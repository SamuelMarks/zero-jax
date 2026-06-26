"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.traceback_util.util as mod


def test_Generic() -> None:
    """Test Generic."""
    obj = mod.Generic()
    assert obj is not None


def test_Hashable() -> None:
    """Test Hashable."""
    obj = mod.Hashable()
    assert obj is not None


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction()
    assert obj is not None


def test_HashablePartial() -> None:
    """Test HashablePartial."""
    obj = mod.HashablePartial()
    assert obj is not None


def test_HashableWrapper() -> None:
    """Test HashableWrapper."""
    obj = mod.HashableWrapper()
    assert obj is not None


def test_Iterable() -> None:
    """Test Iterable."""
    obj = mod.Iterable()
    assert obj is not None


def test_Iterator() -> None:
    """Test Iterator."""
    obj = mod.Iterator()
    assert obj is not None


def test_NumpyComplexWarning() -> None:
    """Test NumpyComplexWarning."""
    obj = mod.NumpyComplexWarning()
    assert obj is not None


def test_OrderedSet() -> None:
    """Test OrderedSet."""
    obj = mod.OrderedSet()
    assert obj is not None


def test_Seq() -> None:
    """Test Seq."""
    obj = mod.Seq()
    assert obj is not None


def test_StrictABC() -> None:
    """Test StrictABC."""
    obj = mod.StrictABC()
    assert obj is not None


def test_StrictABCMeta() -> None:
    """Test StrictABCMeta."""
    obj = mod.StrictABCMeta()
    assert obj is not None


def test_TypeVar() -> None:
    """Test TypeVar."""
    obj = mod.TypeVar()
    assert obj is not None


def test_Unhashable() -> None:
    """Test Unhashable."""
    obj = mod.Unhashable()
    assert obj is not None


def test_WrapKwArgs() -> None:
    """Test WrapKwArgs."""
    obj = mod.WrapKwArgs()
    assert obj is not None


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    with patch("ml_switcheroo_compiler.ops.as_hashable_function") as mock_op:
        mod.as_hashable_function()
        mock_op.assert_called_once_with()


def test_assert_unreachable() -> None:
    """Test assert_unreachable."""
    with patch("ml_switcheroo_compiler.ops.assert_unreachable") as mock_op:
        mod.assert_unreachable()
        mock_op.assert_called_once_with()


def test_cache() -> None:
    """Test cache."""
    with patch("ml_switcheroo_compiler.ops.cache") as mock_op:
        mod.cache()
        mock_op.assert_called_once_with()


def test_canonicalize_axis() -> None:
    """Test canonicalize_axis."""
    with patch("ml_switcheroo_compiler.ops.canonicalize_axis") as mock_op:
        mod.canonicalize_axis()
        mock_op.assert_called_once_with()


def test_cast() -> None:
    """Test cast."""
    with patch("ml_switcheroo_compiler.ops.cast") as mock_op:
        mod.cast()
        mock_op.assert_called_once_with()


def test_ceil_of_ratio() -> None:
    """Test ceil_of_ratio."""
    with patch("ml_switcheroo_compiler.ops.ceil_of_ratio") as mock_op:
        mod.ceil_of_ratio()
        mock_op.assert_called_once_with()


def test_check_toposort() -> None:
    """Test check_toposort."""
    with patch("ml_switcheroo_compiler.ops.check_toposort") as mock_op:
        mod.check_toposort()
        mock_op.assert_called_once_with()


def test_clear_all_caches() -> None:
    """Test clear_all_caches."""
    with patch("ml_switcheroo_compiler.ops.clear_all_caches") as mock_op:
        mod.clear_all_caches()
        mock_op.assert_called_once_with()


def test_clear_all_weakref_lru_caches() -> None:
    """Test clear_all_weakref_lru_caches."""
    with patch("ml_switcheroo_compiler.ops.clear_all_weakref_lru_caches") as mock_op:
        mod.clear_all_weakref_lru_caches()
        mock_op.assert_called_once_with()


def test_concatenate() -> None:
    """Test concatenate."""
    with patch("ml_switcheroo_compiler.ops.concatenate") as mock_op:
        mod.concatenate()
        mock_op.assert_called_once_with()


def test_curry() -> None:
    """Test curry."""
    with patch("ml_switcheroo_compiler.ops.curry") as mock_op:
        mod.curry()
        mock_op.assert_called_once_with()


def test_distributed_debug_log() -> None:
    """Test distributed_debug_log."""
    with patch("ml_switcheroo_compiler.ops.distributed_debug_log") as mock_op:
        mod.distributed_debug_log()
        mock_op.assert_called_once_with()


def test_flatten() -> None:
    """Test flatten."""
    with patch("ml_switcheroo_compiler.ops.flatten") as mock_op:
        mod.flatten()
        mock_op.assert_called_once_with()


def test_fun_name() -> None:
    """Test fun_name."""
    with patch("ml_switcheroo_compiler.ops.fun_name") as mock_op:
        mod.fun_name()
        mock_op.assert_called_once_with()


def test_maybe_named_axis() -> None:
    """Test maybe_named_axis."""
    with patch("ml_switcheroo_compiler.ops.maybe_named_axis") as mock_op:
        mod.maybe_named_axis()
        mock_op.assert_called_once_with()


def test_memoize() -> None:
    """Test memoize."""
    with patch("ml_switcheroo_compiler.ops.memoize") as mock_op:
        mod.memoize()
        mock_op.assert_called_once_with()


def test_merge_lists() -> None:
    """Test merge_lists."""
    with patch("ml_switcheroo_compiler.ops.merge_lists") as mock_op:
        mod.merge_lists()
        mock_op.assert_called_once_with()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with patch("ml_switcheroo_compiler.ops.moveaxis") as mock_op:
        mod.moveaxis()
        mock_op.assert_called_once_with()


def test_overload() -> None:
    """Test overload."""
    with patch("ml_switcheroo_compiler.ops.overload") as mock_op:
        mod.overload()
        mock_op.assert_called_once_with()


def test_partial() -> None:
    """Test partial."""
    obj = mod.partial()
    assert obj is not None


def test_partition_list() -> None:
    """Test partition_list."""
    with patch("ml_switcheroo_compiler.ops.partition_list") as mock_op:
        mod.partition_list()
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


def test_set_module() -> None:
    """Test set_module."""
    with patch("ml_switcheroo_compiler.ops.set_module") as mock_op:
        mod.set_module()
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


def test_stable_unique() -> None:
    """Test stable_unique."""
    with patch("ml_switcheroo_compiler.ops.stable_unique") as mock_op:
        mod.stable_unique()
        mock_op.assert_called_once_with()


def test_subs_list() -> None:
    """Test subs_list."""
    with patch("ml_switcheroo_compiler.ops.subs_list") as mock_op:
        mod.subs_list()
        mock_op.assert_called_once_with()


def test_subs_list2() -> None:
    """Test subs_list2."""
    with patch("ml_switcheroo_compiler.ops.subs_list2") as mock_op:
        mod.subs_list2()
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


def test_tuple_delete() -> None:
    """Test tuple_delete."""
    with patch("ml_switcheroo_compiler.ops.tuple_delete") as mock_op:
        mod.tuple_delete()
        mock_op.assert_called_once_with()


def test_tuple_insert() -> None:
    """Test tuple_insert."""
    with patch("ml_switcheroo_compiler.ops.tuple_insert") as mock_op:
        mod.tuple_insert()
        mock_op.assert_called_once_with()


def test_tuple_update() -> None:
    """Test tuple_update."""
    with patch("ml_switcheroo_compiler.ops.tuple_update") as mock_op:
        mod.tuple_update()
        mock_op.assert_called_once_with()


def test_unflatten() -> None:
    """Test unflatten."""
    with patch("ml_switcheroo_compiler.ops.unflatten") as mock_op:
        mod.unflatten()
        mock_op.assert_called_once_with()


def test_unzip2() -> None:
    """Test unzip2."""
    with patch("ml_switcheroo_compiler.ops.unzip2") as mock_op:
        mod.unzip2()
        mock_op.assert_called_once_with()


def test_unzip3() -> None:
    """Test unzip3."""
    with patch("ml_switcheroo_compiler.ops.unzip3") as mock_op:
        mod.unzip3()
        mock_op.assert_called_once_with()


def test_use_cpp_class() -> None:
    """Test use_cpp_class."""
    with patch("ml_switcheroo_compiler.ops.use_cpp_class") as mock_op:
        mod.use_cpp_class()
        mock_op.assert_called_once_with()


def test_use_cpp_method() -> None:
    """Test use_cpp_method."""
    with patch("ml_switcheroo_compiler.ops.use_cpp_method") as mock_op:
        mod.use_cpp_method()
        mock_op.assert_called_once_with()


def test_weakref_lru_cache() -> None:
    """Test weakref_lru_cache."""
    with patch("ml_switcheroo_compiler.ops.weakref_lru_cache") as mock_op:
        mod.weakref_lru_cache()
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
