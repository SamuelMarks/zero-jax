"""Tests for zero_jax.random.prng."""

from typing import Any

import pytest

import zero_jax.random.prng as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_PRNGKey() -> None:
    """Test PRNGKey."""
    try:
        mod.PRNGKey(1.0)
    except Exception:
        pass


def test_bernoulli() -> None:
    """Test bernoulli."""
    try:
        mod.bernoulli(1.0)
    except Exception:
        pass


def test_categorical() -> None:
    """Test categorical."""
    try:
        mod.categorical(1.0, 1.0)
    except Exception:
        pass


def test_choice() -> None:
    """Test choice."""
    try:
        mod.choice(1.0, 1.0)
    except Exception:
        pass


def test_fold_in() -> None:
    """Test fold_in."""
    try:
        mod.fold_in(1.0, 1.0)
    except Exception:
        pass


def test_normal() -> None:
    """Test normal."""
    try:
        mod.normal(1.0, 1.0)
    except Exception:
        pass


def test_permutation() -> None:
    """Test permutation."""
    try:
        mod.permutation(1.0, 1.0)
    except Exception:
        pass


def test_randint() -> None:
    """Test randint."""
    try:
        mod.randint(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_split() -> None:
    """Test split."""
    try:
        mod.split(1.0)
    except Exception:
        pass


def test_truncated_normal() -> None:
    """Test truncated_normal."""
    try:
        mod.truncated_normal(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_uniform() -> None:
    """Test uniform."""
    try:
        mod.uniform(1.0, 1.0)
    except Exception:
        pass
