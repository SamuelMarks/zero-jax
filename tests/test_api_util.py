"""Tests for zero_jax module."""

import pytest
import zero_jax.api_util as mod


def test_argnums_partial() -> None:
    """Test argnums_partial."""
    with pytest.raises(NotImplementedError):
        mod.argnums_partial()


def test_donation_vector() -> None:
    """Test donation_vector."""
    with pytest.raises(NotImplementedError):
        mod.donation_vector()


def test_flatten_axes() -> None:
    """Test flatten_axes."""
    with pytest.raises(NotImplementedError):
        mod.flatten_axes()


def test_flatten_fun() -> None:
    """Test flatten_fun."""
    with pytest.raises(NotImplementedError):
        mod.flatten_fun()


def test_flatten_fun_nokwargs() -> None:
    """Test flatten_fun_nokwargs."""
    with pytest.raises(NotImplementedError):
        mod.flatten_fun_nokwargs()


def test_rebase_donate_argnums() -> None:
    """Test rebase_donate_argnums."""
    with pytest.raises(NotImplementedError):
        mod.rebase_donate_argnums()


def test_safe_map() -> None:
    """Test safe_map."""
    with pytest.raises(NotImplementedError):
        mod.safe_map()


def test_shaped_abstractify() -> None:
    """Test shaped_abstractify."""
    with pytest.raises(NotImplementedError):
        mod.shaped_abstractify()
