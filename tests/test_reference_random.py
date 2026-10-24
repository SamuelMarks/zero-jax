"""Tests for tests/test_reference_random.py."""

from __future__ import annotations

import jax.random as rand_ref
import numpy as np
import pytest

import zero_jax.random as rand_zero
from zero_jax._types import ArrayLikeSpec


def test_random_skip_split(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_split``."""
    key = rand_ref.PRNGKey(42)
    keys_z = rand_zero.split(key, 2)
    keys_r = rand_ref.split(key, 2)
    assert getattr(keys_z, "shape", None) == getattr(keys_r, "shape", None)


def test_random_skip_fold_in(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_fold_in``."""
    key = rand_ref.PRNGKey(42)
    key_z = rand_zero.fold_in(key, 1)
    key_r = rand_ref.fold_in(key, 1)
    assert getattr(key_z, "shape", None) == getattr(key_r, "shape", None)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_random_skip_uniform(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_uniform``."""
    key = rand_ref.PRNGKey(42)
    z = rand_zero.uniform(key, (2, 2))
    r = rand_ref.uniform(key, (2, 2))
    assert getattr(z, "shape", None) == getattr(r, "shape", None)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_random_skip_normal(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_normal``."""
    key = rand_ref.PRNGKey(42)
    z = rand_zero.normal(key, (2, 2))
    r = rand_ref.normal(key, (2, 2))
    assert getattr(z, "shape", None) == getattr(r, "shape", None)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_random_skip_randint(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_randint``."""
    key = rand_ref.PRNGKey(42)
    z = rand_zero.randint(key, (2, 2), 0, 10)
    r = rand_ref.randint(key, (2, 2), 0, 10)
    assert getattr(z, "shape", None) == getattr(r, "shape", None)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_random_skip_bernoulli(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_bernoulli``."""
    key = rand_ref.PRNGKey(42)
    z = rand_zero.bernoulli(key, 0.5, (2, 2))
    r = rand_ref.bernoulli(key, 0.5, (2, 2))
    assert getattr(z, "shape", None) == getattr(r, "shape", None)


def test_random_skip_categorical(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_categorical``."""
    key = rand_ref.PRNGKey(42)
    logits = np.array([0.1, 0.2, 0.7])
    z = rand_zero.categorical(key, logits)
    r = rand_ref.categorical(key, logits)
    assert getattr(z, "shape", None) == getattr(r, "shape", None)


def test_random_skip_permutation(check_allclose: ArrayLikeSpec) -> None:
    """Execute ``test_random_skip_permutation``."""
    key = rand_ref.PRNGKey(42)
    x = np.arange(10)
    z = rand_zero.permutation(key, x)
    r = rand_ref.permutation(key, x)
    assert getattr(z, "shape", None) == getattr(r, "shape", None)
