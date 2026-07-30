"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.traceback_util.util as mod


def test_Generic() -> None:
    """Test Generic."""
    obj = mod.Generic(id=1, name="test_Generic", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Generic"
    assert obj.value == "test_value"


def test_Hashable() -> None:
    """Test Hashable."""
    obj = mod.Hashable(id=1, name="test_Hashable", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Hashable"
    assert obj.value == "test_value"


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction(id=1, name="test_HashableFunction", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_HashableFunction"
    assert obj.value == "test_value"


def test_HashablePartial() -> None:
    """Test HashablePartial."""
    obj = mod.HashablePartial(id=1, name="test_HashablePartial", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_HashablePartial"
    assert obj.value == "test_value"


def test_HashableWrapper() -> None:
    """Test HashableWrapper."""
    obj = mod.HashableWrapper(id=1, name="test_HashableWrapper", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_HashableWrapper"
    assert obj.value == "test_value"


def test_Iterable() -> None:
    """Test Iterable."""
    obj = mod.Iterable(id=1, name="test_Iterable", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Iterable"
    assert obj.value == "test_value"


def test_Iterator() -> None:
    """Test Iterator."""
    obj = mod.Iterator(id=1, name="test_Iterator", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Iterator"
    assert obj.value == "test_value"


def test_NumpyComplexWarning() -> None:
    """Test NumpyComplexWarning."""
    obj = mod.NumpyComplexWarning(
        id=1, name="test_NumpyComplexWarning", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_NumpyComplexWarning"
    assert obj.value == "test_value"


def test_OrderedSet() -> None:
    """Test OrderedSet."""
    obj = mod.OrderedSet(id=1, name="test_OrderedSet", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_OrderedSet"
    assert obj.value == "test_value"


def test_Seq() -> None:
    """Test Seq."""
    obj = mod.Seq(id=1, name="test_Seq", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Seq"
    assert obj.value == "test_value"


def test_StrictABC() -> None:
    """Test StrictABC."""
    obj = mod.StrictABC(id=1, name="test_StrictABC", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_StrictABC"
    assert obj.value == "test_value"


def test_StrictABCMeta() -> None:
    """Test StrictABCMeta."""
    obj = mod.StrictABCMeta(id=1, name="test_StrictABCMeta", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_StrictABCMeta"
    assert obj.value == "test_value"


def test_TypeVar() -> None:
    """Test TypeVar."""
    obj = mod.TypeVar(id=1, name="test_TypeVar", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_TypeVar"
    assert obj.value == "test_value"


def test_Unhashable() -> None:
    """Test Unhashable."""
    obj = mod.Unhashable(id=1, name="test_Unhashable", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Unhashable"
    assert obj.value == "test_value"


def test_WrapKwArgs() -> None:
    """Test WrapKwArgs."""
    obj = mod.WrapKwArgs(id=1, name="test_WrapKwArgs", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_WrapKwArgs"
    assert obj.value == "test_value"


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    with patch(
        "zero_jax._compiler_proxy_ops.as_hashable_function", create=True
    ) as mock_op:
        mod.as_hashable_function()
        mock_op.assert_called_once_with()


def test_assert_unreachable() -> None:
    """Test assert_unreachable."""
    with patch(
        "zero_jax._compiler_proxy_ops.assert_unreachable", create=True
    ) as mock_op:
        mod.assert_unreachable()
        mock_op.assert_called_once_with()


def test_cache() -> None:
    """Test cache."""
    with patch("zero_jax._compiler_proxy_ops.cache", create=True) as mock_op:
        mod.cache()
        mock_op.assert_called_once_with()


def test_canonicalize_axis() -> None:
    """Test canonicalize_axis."""
    with patch(
        "zero_jax._compiler_proxy_ops.canonicalize_axis", create=True
    ) as mock_op:
        mod.canonicalize_axis()
        mock_op.assert_called_once_with()


def test_cast() -> None:
    """Test cast."""
    with patch("zero_jax._compiler_proxy_ops.cast", create=True) as mock_op:
        mod.cast()
        mock_op.assert_called_once_with()


def test_ceil_of_ratio() -> None:
    """Test ceil_of_ratio."""
    with patch("zero_jax._compiler_proxy_ops.ceil_of_ratio", create=True) as mock_op:
        mod.ceil_of_ratio()
        mock_op.assert_called_once_with()


def test_check_toposort() -> None:
    """Test check_toposort."""
    with patch("zero_jax._compiler_proxy_ops.check_toposort", create=True) as mock_op:
        mod.check_toposort()
        mock_op.assert_called_once_with()


def test_clear_all_caches() -> None:
    """Test clear_all_caches."""
    with patch("zero_jax._compiler_proxy_ops.clear_all_caches", create=True) as mock_op:
        mod.clear_all_caches()
        mock_op.assert_called_once_with()


def test_clear_all_weakref_lru_caches() -> None:
    """Test clear_all_weakref_lru_caches."""
    with patch(
        "zero_jax._compiler_proxy_ops.clear_all_weakref_lru_caches", create=True
    ) as mock_op:
        mod.clear_all_weakref_lru_caches()
        mock_op.assert_called_once_with()


def test_concatenate() -> None:
    """Test concatenate."""
    with patch("zero_jax._compiler_proxy_ops.concatenate", create=True) as mock_op:
        mod.concatenate()
        mock_op.assert_called_once_with()


def test_curry() -> None:
    """Test curry."""
    with patch("zero_jax._compiler_proxy_ops.curry", create=True) as mock_op:
        mod.curry()
        mock_op.assert_called_once_with()


def test_distributed_debug_log() -> None:
    """Test distributed_debug_log."""
    with patch(
        "zero_jax._compiler_proxy_ops.distributed_debug_log", create=True
    ) as mock_op:
        mod.distributed_debug_log()
        mock_op.assert_called_once_with()


def test_flatten() -> None:
    """Test flatten."""
    with patch("zero_jax._compiler_proxy_ops.flatten", create=True) as mock_op:
        mod.flatten()
        mock_op.assert_called_once_with()


def test_fun_name() -> None:
    """Test fun_name."""
    with patch("zero_jax._compiler_proxy_ops.fun_name", create=True) as mock_op:
        mod.fun_name()
        mock_op.assert_called_once_with()


def test_maybe_named_axis() -> None:
    """Test maybe_named_axis."""
    with patch("zero_jax._compiler_proxy_ops.maybe_named_axis", create=True) as mock_op:
        mod.maybe_named_axis()
        mock_op.assert_called_once_with()


def test_memoize() -> None:
    """Test memoize."""
    with patch("zero_jax._compiler_proxy_ops.memoize", create=True) as mock_op:
        mod.memoize()
        mock_op.assert_called_once_with()


def test_merge_lists() -> None:
    """Test merge_lists."""
    with patch("zero_jax._compiler_proxy_ops.merge_lists", create=True) as mock_op:
        mod.merge_lists()
        mock_op.assert_called_once_with()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with patch("zero_jax._compiler_proxy_ops.moveaxis", create=True) as mock_op:
        mod.moveaxis()
        mock_op.assert_called_once_with()


def test_overload() -> None:
    """Test overload."""
    with patch("zero_jax._compiler_proxy_ops.overload", create=True) as mock_op:
        mod.overload()
        mock_op.assert_called_once_with()


def test_partial() -> None:
    """Test partial."""
    obj = mod.partial(id=1, name="test_partial", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_partial"
    assert obj.value == "test_value"


def test_partition_list() -> None:
    """Test partition_list."""
    with patch("zero_jax._compiler_proxy_ops.partition_list", create=True) as mock_op:
        mod.partition_list()
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


def test_set_module() -> None:
    """Test set_module."""
    with patch("zero_jax._compiler_proxy_ops.set_module", create=True) as mock_op:
        mod.set_module()
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


def test_stable_unique() -> None:
    """Test stable_unique."""
    with patch("zero_jax._compiler_proxy_ops.stable_unique", create=True) as mock_op:
        mod.stable_unique()
        mock_op.assert_called_once_with()


def test_subs_list() -> None:
    """Test subs_list."""
    with patch("zero_jax._compiler_proxy_ops.subs_list", create=True) as mock_op:
        mod.subs_list()
        mock_op.assert_called_once_with()


def test_subs_list2() -> None:
    """Test subs_list2."""
    with patch("zero_jax._compiler_proxy_ops.subs_list2", create=True) as mock_op:
        mod.subs_list2()
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


def test_tuple_delete() -> None:
    """Test tuple_delete."""
    with patch("zero_jax._compiler_proxy_ops.tuple_delete", create=True) as mock_op:
        mod.tuple_delete()
        mock_op.assert_called_once_with()


def test_tuple_insert() -> None:
    """Test tuple_insert."""
    with patch("zero_jax._compiler_proxy_ops.tuple_insert", create=True) as mock_op:
        mod.tuple_insert()
        mock_op.assert_called_once_with()


def test_tuple_update() -> None:
    """Test tuple_update."""
    with patch("zero_jax._compiler_proxy_ops.tuple_update", create=True) as mock_op:
        mod.tuple_update()
        mock_op.assert_called_once_with()


def test_unflatten() -> None:
    """Test unflatten."""
    with patch("zero_jax._compiler_proxy_ops.unflatten", create=True) as mock_op:
        mod.unflatten()
        mock_op.assert_called_once_with()


def test_unzip2() -> None:
    """Test unzip2."""
    with patch("zero_jax._compiler_proxy_ops.unzip2", create=True) as mock_op:
        mod.unzip2()
        mock_op.assert_called_once_with()


def test_unzip3() -> None:
    """Test unzip3."""
    with patch("zero_jax._compiler_proxy_ops.unzip3", create=True) as mock_op:
        mod.unzip3()
        mock_op.assert_called_once_with()


def test_use_cpp_class() -> None:
    """Test use_cpp_class."""
    with patch("zero_jax._compiler_proxy_ops.use_cpp_class", create=True) as mock_op:
        mod.use_cpp_class()
        mock_op.assert_called_once_with()


def test_use_cpp_method() -> None:
    """Test use_cpp_method."""
    with patch("zero_jax._compiler_proxy_ops.use_cpp_method", create=True) as mock_op:
        mod.use_cpp_method()
        mock_op.assert_called_once_with()


def test_weakref_lru_cache() -> None:
    """Test weakref_lru_cache."""
    with patch(
        "zero_jax._compiler_proxy_ops.weakref_lru_cache", create=True
    ) as mock_op:
        mod.weakref_lru_cache()
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


def test_typing_vars() -> None:
    """Test typing vars."""
    assert mod.T1 is not None
    assert mod.T2 is not None
    assert mod.T3 is not None
    assert hasattr(mod, "TYPE_CHECKING")
    assert hasattr(mod, "Sequence")
    assert isinstance(mod.cache_clearing_funs, list)
