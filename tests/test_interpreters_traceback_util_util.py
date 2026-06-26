"""Tests for zero_jax module."""

import pytest
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
    with pytest.raises(NotImplementedError):
        mod.as_hashable_function()


def test_assert_unreachable() -> None:
    """Test assert_unreachable."""
    with pytest.raises(NotImplementedError):
        mod.assert_unreachable()


def test_cache() -> None:
    """Test cache."""
    with pytest.raises(NotImplementedError):
        mod.cache()


def test_canonicalize_axis() -> None:
    """Test canonicalize_axis."""
    with pytest.raises(NotImplementedError):
        mod.canonicalize_axis()


def test_cast() -> None:
    """Test cast."""
    with pytest.raises(NotImplementedError):
        mod.cast()


def test_ceil_of_ratio() -> None:
    """Test ceil_of_ratio."""
    with pytest.raises(NotImplementedError):
        mod.ceil_of_ratio()


def test_check_toposort() -> None:
    """Test check_toposort."""
    with pytest.raises(NotImplementedError):
        mod.check_toposort()


def test_clear_all_caches() -> None:
    """Test clear_all_caches."""
    with pytest.raises(NotImplementedError):
        mod.clear_all_caches()


def test_clear_all_weakref_lru_caches() -> None:
    """Test clear_all_weakref_lru_caches."""
    with pytest.raises(NotImplementedError):
        mod.clear_all_weakref_lru_caches()


def test_concatenate() -> None:
    """Test concatenate."""
    with pytest.raises(NotImplementedError):
        mod.concatenate()


def test_curry() -> None:
    """Test curry."""
    with pytest.raises(NotImplementedError):
        mod.curry()


def test_distributed_debug_log() -> None:
    """Test distributed_debug_log."""
    with pytest.raises(NotImplementedError):
        mod.distributed_debug_log()


def test_flatten() -> None:
    """Test flatten."""
    with pytest.raises(NotImplementedError):
        mod.flatten()


def test_fun_name() -> None:
    """Test fun_name."""
    with pytest.raises(NotImplementedError):
        mod.fun_name()


def test_maybe_named_axis() -> None:
    """Test maybe_named_axis."""
    with pytest.raises(NotImplementedError):
        mod.maybe_named_axis()


def test_memoize() -> None:
    """Test memoize."""
    with pytest.raises(NotImplementedError):
        mod.memoize()


def test_merge_lists() -> None:
    """Test merge_lists."""
    with pytest.raises(NotImplementedError):
        mod.merge_lists()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with pytest.raises(NotImplementedError):
        mod.moveaxis()


def test_overload() -> None:
    """Test overload."""
    with pytest.raises(NotImplementedError):
        mod.overload()


def test_partial() -> None:
    """Test partial."""
    obj = mod.partial()
    assert obj is not None


def test_partition_list() -> None:
    """Test partition_list."""
    with pytest.raises(NotImplementedError):
        mod.partition_list()


def test_safe_map() -> None:
    """Test safe_map."""
    with pytest.raises(NotImplementedError):
        mod.safe_map()


def test_safe_zip() -> None:
    """Test safe_zip."""
    with pytest.raises(NotImplementedError):
        mod.safe_zip()


def test_set_module() -> None:
    """Test set_module."""
    with pytest.raises(NotImplementedError):
        mod.set_module()


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


def test_stable_unique() -> None:
    """Test stable_unique."""
    with pytest.raises(NotImplementedError):
        mod.stable_unique()


def test_subs_list() -> None:
    """Test subs_list."""
    with pytest.raises(NotImplementedError):
        mod.subs_list()


def test_subs_list2() -> None:
    """Test subs_list2."""
    with pytest.raises(NotImplementedError):
        mod.subs_list2()


def test_subvals() -> None:
    """Test subvals."""
    with pytest.raises(NotImplementedError):
        mod.subvals()


def test_toposort() -> None:
    """Test toposort."""
    with pytest.raises(NotImplementedError):
        mod.toposort()


def test_tuple_delete() -> None:
    """Test tuple_delete."""
    with pytest.raises(NotImplementedError):
        mod.tuple_delete()


def test_tuple_insert() -> None:
    """Test tuple_insert."""
    with pytest.raises(NotImplementedError):
        mod.tuple_insert()


def test_tuple_update() -> None:
    """Test tuple_update."""
    with pytest.raises(NotImplementedError):
        mod.tuple_update()


def test_unflatten() -> None:
    """Test unflatten."""
    with pytest.raises(NotImplementedError):
        mod.unflatten()


def test_unzip2() -> None:
    """Test unzip2."""
    with pytest.raises(NotImplementedError):
        mod.unzip2()


def test_unzip3() -> None:
    """Test unzip3."""
    with pytest.raises(NotImplementedError):
        mod.unzip3()


def test_use_cpp_class() -> None:
    """Test use_cpp_class."""
    with pytest.raises(NotImplementedError):
        mod.use_cpp_class()


def test_use_cpp_method() -> None:
    """Test use_cpp_method."""
    with pytest.raises(NotImplementedError):
        mod.use_cpp_method()


def test_weakref_lru_cache() -> None:
    """Test weakref_lru_cache."""
    with pytest.raises(NotImplementedError):
        mod.weakref_lru_cache()


def test_wrap_name() -> None:
    """Test wrap_name."""
    with pytest.raises(NotImplementedError):
        mod.wrap_name()


def test_wraps() -> None:
    """Test wraps."""
    with pytest.raises(NotImplementedError):
        mod.wraps()
