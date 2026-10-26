"""Tests for zero_jax.util."""

from __future__ import annotations

import pytest

import zero_jax.util as mod


def test_HashableFunction() -> None:
    """Test HashableFunction."""
    obj = mod.HashableFunction()
    assert obj() is None
    fn = lambda x: x + 1
    hf1 = mod.HashableFunction(fn, (1, 2))
    hf2 = mod.HashableFunction(fn, (1, 2))
    assert hf1 == hf2
    assert hash(hf1) == hash(hf2)
    assert hf1(4) == 5


def test_as_hashable_function() -> None:
    """Test as_hashable_function."""
    hf = mod.as_hashable_function(lambda x: x * 3, ("k",))
    assert hf(4) == 12


def test_cache() -> None:
    """Test cache."""
    calls: list[int] = []

    def fn(x: int) -> int:
        """Compute ``x + 1`` while recording invocations.

        Args:
            x: Input integer.

        Returns:
            int: Incremented integer.
        """
        calls.append(x)
        return x + 1

    cached = mod.cache(fn)
    assert cached(2) == 3
    assert cached(2) == 3
    assert calls == [2]

    sized = mod.cache(128)(fn)
    assert sized(5) == 6


def test_safe_map() -> None:
    """Test safe_map."""
    assert mod.safe_map(lambda a, b: a + b, [1, 2], [3, 4]) == [4, 6]
    assert mod.safe_map(lambda a: a) == []
    with pytest.raises(ValueError):
        mod.safe_map(lambda a, b: a + b, [1], [1, 2])


def test_safe_zip() -> None:
    """Test safe_zip."""
    assert mod.safe_zip([1, 2], [3, 4]) == [(1, 3), (2, 4)]
    assert mod.safe_zip() == []
    with pytest.raises(ValueError):
        mod.safe_zip([1], [1, 2])


def test_split_dict() -> None:
    """Test split_dict."""
    assert mod.split_dict({"a": 10, "b": 20}, [["a"], ["b"]]) == ([10], [20])


def test_split_list() -> None:
    """Test split_list."""
    assert mod.split_list([1, 2, 3, 4], [1, 2]) == [[1], [2, 3], [4]]


def test_split_list_checked() -> None:
    """Test split_list_checked."""
    assert mod.split_list_checked([1, 2, 3], [1, 2]) == [[1], [2, 3]]


def test_split_merge() -> None:
    """Test split_merge."""
    lhs, rhs, merge = mod.split_merge([True, False, True], [10, 20, 30])
    assert lhs == [10, 30]
    assert rhs == [20]
    assert merge(lhs, rhs) == [10, 30, 20]


def test_subvals() -> None:
    """Test subvals."""
    assert mod.subvals((10, 20, 30), [(1, 99)]) == (10, 99, 30)


def test_toposort() -> None:
    """Test toposort."""
    assert mod.toposort([1, 2, 3]) == [1, 2, 3]


def test_unzip2() -> None:
    """Test unzip2."""
    assert mod.unzip2([(1, "a"), (2, "b")]) == ((1, 2), ("a", "b"))


def test_wrap_name() -> None:
    """Test wrap_name."""
    assert mod.wrap_name("f", "vmap") == "vmap(f)"


def test_wraps() -> None:
    """Test wraps."""

    def orig(x: int) -> int:
        """Original function docstring.

        Args:
            x: Input integer.

        Returns:
            int: Output integer.
        """
        return x

    assert orig(5) == 5
    wrapped = mod.wraps(orig)(lambda x: x + 1)
    assert wrapped(2) == 3
    assert wrapped.__name__ == "orig"
