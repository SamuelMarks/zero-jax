"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.api_util as mod


def test_argnums_partial() -> None:
    """Test argnums_partial."""
    with patch("zero_jax._compiler_proxy_ops.argnums_partial", create=True) as mock_op:
        mod.argnums_partial()
        mock_op.assert_called_once_with()


def test_donation_vector() -> None:
    """Test donation_vector."""
    with patch("zero_jax._compiler_proxy_ops.donation_vector", create=True) as mock_op:
        mod.donation_vector()
        mock_op.assert_called_once_with()


def test_flatten_axes() -> None:
    """Test flatten_axes."""
    with patch("zero_jax._compiler_proxy_ops.flatten_axes", create=True) as mock_op:
        mod.flatten_axes()
        mock_op.assert_called_once_with()


def test_flatten_fun() -> None:
    """Test flatten_fun."""
    with patch("zero_jax._compiler_proxy_ops.flatten_fun", create=True) as mock_op:
        mod.flatten_fun()
        mock_op.assert_called_once_with()


def test_flatten_fun_nokwargs() -> None:
    """Test flatten_fun_nokwargs."""
    with patch(
        "zero_jax._compiler_proxy_ops.flatten_fun_nokwargs", create=True
    ) as mock_op:
        mod.flatten_fun_nokwargs()
        mock_op.assert_called_once_with()


def test_rebase_donate_argnums() -> None:
    """Test rebase_donate_argnums."""
    with patch(
        "zero_jax._compiler_proxy_ops.rebase_donate_argnums", create=True
    ) as mock_op:
        mod.rebase_donate_argnums()
        mock_op.assert_called_once_with()


def test_safe_map() -> None:
    """Test safe_map."""
    with patch("zero_jax._compiler_proxy_ops.safe_map", create=True) as mock_op:
        mod.safe_map()
        mock_op.assert_called_once_with()


def test_shaped_abstractify() -> None:
    """Test shaped_abstractify."""
    with patch(
        "zero_jax._compiler_proxy_ops.shaped_abstractify", create=True
    ) as mock_op:
        mod.shaped_abstractify()
        mock_op.assert_called_once_with()
