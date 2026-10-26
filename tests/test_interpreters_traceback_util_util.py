"""Tests for zero_jax.interpreters.traceback_util.util."""

from __future__ import annotations

import pytest

import zero_jax.interpreters.traceback_util.util as mod
import zero_jax.numpy as jnp


def test_Generic() -> None:
    """Test Generic."""
    assert mod.Generic is not None


def test_Hashable() -> None:
    """Test Hashable."""
    assert isinstance(1, mod.Hashable)


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    fn = lambda x: x + 1
    obj = mod.HashableFunction(fn, (1,))
    assert obj(2) == 3
    assert obj == mod.HashableFunction(fn, (1,))
    assert hash(obj) == hash(mod.HashableFunction(fn, (1,)))


def test_HashablePartial() -> None:
    """Test HashablePartial."""
    obj = mod.HashablePartial(id=1, name="test_HashablePartial", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_HashablePartial"
    assert obj.value == "test_value"


def test_HashableWrapper() -> None:
    """Test HashableWrapper."""
    obj = mod.HashableWrapper(id=1, name="test_HashableWrapper", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_HashableWrapper"
    assert obj.value == "test_value"


def test_Iterable() -> None:
    """Test Iterable."""
    assert isinstance([1, 2], mod.Iterable)


def test_Iterator() -> None:
    """Test Iterator."""
    assert isinstance(iter([1, 2]), mod.Iterator)


def test_NumpyComplexWarning() -> None:
    """Test NumpyComplexWarning."""
    w = mod.NumpyComplexWarning("complex cast")
    assert isinstance(w, Warning)


def test_OrderedSet() -> None:
    """Test OrderedSet."""
    obj = mod.OrderedSet(id=1, name="test_OrderedSet", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_OrderedSet"
    assert obj.value == "test_value"


def test_Seq() -> None:
    """Test Seq."""
    s = mod.Seq((1, 2, 3))
    assert tuple(s) == (1, 2, 3)


def test_StrictABC() -> None:
    """Test StrictABC."""
    obj = mod.StrictABC(id=1, name="test_StrictABC", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_StrictABC"
    assert obj.value == "test_value"


def test_StrictABCMeta() -> None:
    """Test StrictABCMeta."""
    assert issubclass(mod.StrictABCMeta, type)


def test_TypeVar() -> None:
    """Test TypeVar."""
    tv = mod.TypeVar("T")
    assert tv.__name__ == "T"


def test_Unhashable() -> None:
    """Test Unhashable."""
    obj = mod.Unhashable(id=1, name="test_Unhashable", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_Unhashable"
    assert obj.value == "test_value"


def test_WrapKwArgs() -> None:
    """Test WrapKwArgs."""
    obj = mod.WrapKwArgs(id=1, name="test_WrapKwArgs", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_WrapKwArgs"
    assert obj.value == "test_value"


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    hf = mod.as_hashable_function(lambda x: x * 2, ("mul2",))
    assert hf(4) == 8


def test_assert_unreachable() -> None:
    """Test assert_unreachable."""
    with pytest.raises(AssertionError):
        mod.assert_unreachable("bad branch")


def test_cache() -> None:
    """Test cache."""
    cached = mod.cache(lambda x: x + 10)
    assert cached(5) == 15


def test_canonicalize_axis() -> None:
    """Test canonicalize_axis."""
    assert mod.canonicalize_axis(-1, 3) == 2
    assert mod.canonicalize_axis(1, 3) == 1
    with pytest.raises(ValueError):
        mod.canonicalize_axis(5, 3)


def test_cast() -> None:
    """Test cast."""
    assert mod.cast(int, 7) == 7
    assert mod.cast(int) is int


def test_ceil_of_ratio() -> None:
    """Test ceil_of_ratio."""
    assert mod.ceil_of_ratio(7, 3) == 3
    assert mod.ceil_of_ratio(6, 3) == 2


def test_check_toposort() -> None:
    """Test check_toposort."""
    assert mod.check_toposort([1, 2, 3]) is True


def test_clear_all_caches() -> None:
    """Test clear_all_caches."""
    called = []
    mod.cache_clearing_funs.append(lambda: called.append(True))
    try:
        assert mod.clear_all_caches() >= 1
        assert called == [True]
    finally:
        mod.cache_clearing_funs.clear()


def test_clear_all_weakref_lru_caches() -> None:
    """Test clear_all_weakref_lru_caches."""
    assert mod.clear_all_weakref_lru_caches() == 0


def test_concatenate() -> None:
    """Test concatenate."""
    assert mod.concatenate([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_curry() -> None:
    """Test curry."""
    c_add = mod.curry(lambda x, y: x + y)
    assert c_add(2)(3) == 5
    assert mod.curry(None) is None


def test_distributed_debug_log() -> None:
    """Test distributed_debug_log."""
    res = mod.distributed_debug_log(("step", 1))
    assert res == [("step", 1)]


def test_flatten() -> None:
    """Test flatten."""
    assert mod.flatten([[1], [2, 3]]) == [1, 2, 3]


def test_fun_name() -> None:
    """Test fun_name."""

    def sample_fn() -> None:
        """Sample function."""

    assert mod.fun_name(sample_fn) == "sample_fn"
    assert mod.fun_name(mod.partial(sample_fn)) == "sample_fn"
    assert mod.fun_name(None) == "<anonymous>"


def test_maybe_named_axis() -> None:
    """Test maybe_named_axis."""
    assert mod.maybe_named_axis(2, lambda i: i + 1, lambda n: str(n)) == 3
    assert mod.maybe_named_axis("batch", lambda i: i, lambda n: f"axis:{n}") == "axis:batch"


def test_memoize() -> None:
    """Test memoize."""
    fn = mod.memoize(lambda x: x * 3)
    assert fn(4) == 12


def test_merge_lists() -> None:
    """Test merge_lists."""
    assert mod.merge_lists([False, True, False], [10, 20], [99]) == [10, 99, 20]


def test_moveaxis() -> None:
    """Test moveaxis."""
    arr = mod.moveaxis(jnp.ones((2, 3)), 0, 1)
    assert arr.shape == (3, 2)


def test_overload() -> None:
    """Test overload."""
    fn = lambda x: x
    assert mod.overload(fn) is fn


def test_partial() -> None:
    """Test partial."""
    p = mod.partial(lambda a, b: a - b, 10)
    assert p(3) == 7


def test_partition_list() -> None:
    """Test partition_list."""
    f_items, t_items = mod.partition_list([False, True, False], [1, 2, 3])
    assert f_items == [1, 3]
    assert t_items == [2]


def test_safe_map() -> None:
    """Test safe_map."""
    assert mod.safe_map(lambda x, y: x + y, [1, 2], [3, 4]) == [4, 6]


def test_safe_zip() -> None:
    """Test safe_zip."""
    assert mod.safe_zip([1, 2], [3, 4]) == [(1, 3), (2, 4)]


def test_set_module() -> None:
    """Test set_module."""
    decorator = mod.set_module("jax")
    fn = lambda x: x
    assert decorator(fn) is fn


def test_split_dict() -> None:
    """Test split_dict."""
    assert mod.split_dict({"a": 1, "b": 2}, [["a"], ["b"]]) == ([1], [2])


def test_split_list() -> None:
    """Test split_list."""
    assert mod.split_list([1, 2, 3, 4], [2]) == [[1, 2], [3, 4]]


def test_split_list_checked() -> None:
    """Test split_list_checked."""
    assert mod.split_list_checked([1, 2, 3], [1, 2]) == [[1], [2, 3]]


def test_split_merge() -> None:
    """Test split_merge."""
    lhs, rhs, merge = mod.split_merge([True, False], [10, 20])
    assert merge(lhs, rhs) == [10, 20]


def test_stable_unique() -> None:
    """Test stable_unique."""
    assert mod.stable_unique([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_subs_list() -> None:
    """Test subs_list."""
    assert mod.subs_list(["a", "b"], [1, 2], [1, 3, 2]) == ["a", 3, "b"]


def test_subs_list2() -> None:
    """Test subs_list2."""
    assert mod.subs_list2(["a"], ["b"], [1], [1, 2]) == ["a", 2]


def test_subvals() -> None:
    """Test subvals."""
    assert mod.subvals((10, 20, 30), [(1, 99)]) == (10, 99, 30)


def test_toposort() -> None:
    """Test toposort."""
    assert mod.toposort([1, 2]) == [1, 2]


def test_tuple_delete() -> None:
    """Test tuple_delete."""
    assert mod.tuple_delete((10, 20, 30), 1) == (10, 30)


def test_tuple_insert() -> None:
    """Test tuple_insert."""
    assert mod.tuple_insert((10, 30), 1, 20) == (10, 20, 30)


def test_tuple_update() -> None:
    """Test tuple_update."""
    assert mod.tuple_update((10, 20, 30), 1, 99) == (10, 99, 30)
    assert mod.tuple_update((), 0, 5) == (5,)


def test_unflatten() -> None:
    """Test unflatten."""
    assert mod.unflatten([1, 2, 3, 4], [2, 2]) == [[1, 2], [3, 4]]
    assert mod.unflatten([], []) == []


def test_unzip2() -> None:
    """Test unzip2."""
    assert mod.unzip2([(1, "a"), (2, "b")]) == ((1, 2), ("a", "b"))


def test_unzip3() -> None:
    """Test unzip3."""
    assert mod.unzip3([(1, "a", True), (2, "b", False)]) == ((1, 2), ("a", "b"), (True, False))


def test_use_cpp_class() -> None:
    """Test use_cpp_class."""
    cls_dec = mod.use_cpp_class(None)
    assert cls_dec(int) is int


def test_use_cpp_method() -> None:
    """Test use_cpp_method."""
    meth_dec = mod.use_cpp_method(True)
    fn = lambda self: None
    assert meth_dec(fn) is fn


def test_weakref_lru_cache() -> None:
    """Test weakref_lru_cache."""
    cached = mod.weakref_lru_cache(lambda x: x + 1)
    assert cached(3) == 4
    cached2 = mod.weakref_lru_cache(maxsize=128)(lambda x: x * 2)
    assert cached2(3) == 6


def test_wrap_name() -> None:
    """Test wrap_name."""
    assert mod.wrap_name("fn", "jit") == "jit(fn)"


def test_wraps() -> None:
    """Test wraps."""

    def orig(x: int) -> int:
        """Original function.

        Args:
            x: Input integer.

        Returns:
            int: Unmodified integer.
        """
        return x

    assert orig(2) == 2
    wrapped = mod.wraps(orig)(lambda x: x + 1)
    assert wrapped(2) == 3


def test_typing_vars() -> None:
    """Test typing vars."""
    assert mod.T1 is not None
    assert mod.T2 is not None
    assert mod.T3 is not None
    assert "TYPE_CHECKING" in dir(mod)
    assert mod.Sequence is not None
    assert isinstance(mod.cache_clearing_funs, list)
