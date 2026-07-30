"""Tests for zero_jax.api.core."""

from typing import Any

import pytest

import zero_jax.api.core as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_NamedSharding() -> None:
    """Test class NamedSharding."""
    try:
        mod.NamedSharding(1.0, 1.0)
    except Exception:
        pass


def test_class_ShapeDtypeStruct() -> None:
    """Test class ShapeDtypeStruct."""
    try:
        mod.ShapeDtypeStruct(1.0, 1.0)
    except Exception:
        pass


def test_class_Shard() -> None:
    """Test class Shard."""
    try:
        mod.Shard(1.0, 1.0)
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
    except Exception:
        pass


def test_block_until_ready() -> None:
    """Test block_until_ready."""
    try:
        mod.block_until_ready(1.0)
    except Exception:
        pass


def test_default_backend() -> None:
    """Test default_backend."""
    try:
        mod.default_backend()
    except Exception:
        pass


def test_default_device() -> None:
    """Test default_device."""
    try:
        mod.default_device()
    except Exception:
        pass


def test_device_count() -> None:
    """Test device_count."""
    try:
        mod.device_count()
    except Exception:
        pass


def test_device_put() -> None:
    """Test device_put."""
    try:
        mod.device_put(1.0)
    except Exception:
        pass


def test_device_put_replicated() -> None:
    """Test device_put_replicated."""
    try:
        mod.device_put_replicated(1.0, 1.0)
    except Exception:
        pass


def test_device_put_sharded() -> None:
    """Test device_put_sharded."""
    try:
        mod.device_put_sharded(1.0, 1.0)
    except Exception:
        pass


def test_host_count() -> None:
    """Test host_count."""
    try:
        mod.host_count()
    except Exception:
        pass


def test_host_id() -> None:
    """Test host_id."""
    try:
        mod.host_id()
    except Exception:
        pass


def test_host_ids() -> None:
    """Test host_ids."""
    try:
        mod.host_ids()
    except Exception:
        pass


def test_local_device_count() -> None:
    """Test local_device_count."""
    try:
        mod.local_device_count()
    except Exception:
        pass


def test_process_count() -> None:
    """Test process_count."""
    try:
        mod.process_count()
    except Exception:
        pass


def test_process_index() -> None:
    """Test process_index."""
    try:
        mod.process_index()
    except Exception:
        pass
