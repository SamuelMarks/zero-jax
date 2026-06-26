"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.xla.xb as mod


def test_BackendFactory() -> None:
    """Test BackendFactory."""
    with patch("ml_switcheroo_compiler.ops.BackendFactory") as mock_op:
        mod.BackendFactory()
        mock_op.assert_called_once_with()


def test_BackendRegistration() -> None:
    """Test BackendRegistration."""
    obj = mod.BackendRegistration()
    assert obj is not None


def test_Mapping() -> None:
    """Test Mapping."""
    obj = mod.Mapping()
    assert obj is not None


def test_TopologyFactory() -> None:
    """Test TopologyFactory."""
    with patch("ml_switcheroo_compiler.ops.TopologyFactory") as mock_op:
        mod.TopologyFactory()
        mock_op.assert_called_once_with()


def test_XlaBackend() -> None:
    """Test XlaBackend."""
    obj = mod.XlaBackend()
    assert obj is not None


def test_backend_pjrt_c_api_version() -> None:
    """Test backend_pjrt_c_api_version."""
    with patch("ml_switcheroo_compiler.ops.backend_pjrt_c_api_version") as mock_op:
        mod.backend_pjrt_c_api_version()
        mock_op.assert_called_once_with()


def test_backend_xla_version() -> None:
    """Test backend_xla_version."""
    with patch("ml_switcheroo_compiler.ops.backend_xla_version") as mock_op:
        mod.backend_xla_version()
        mock_op.assert_called_once_with()


def test_backends() -> None:
    """Test backends."""
    with patch("ml_switcheroo_compiler.ops.backends") as mock_op:
        mod.backends()
        mock_op.assert_called_once_with()


def test_backends_are_initialized() -> None:
    """Test backends_are_initialized."""
    with patch("ml_switcheroo_compiler.ops.backends_are_initialized") as mock_op:
        mod.backends_are_initialized()
        mock_op.assert_called_once_with()


def test_canonicalize_platform() -> None:
    """Test canonicalize_platform."""
    with patch("ml_switcheroo_compiler.ops.canonicalize_platform") as mock_op:
        mod.canonicalize_platform()
        mock_op.assert_called_once_with()


def test_default_backend() -> None:
    """Test default_backend."""
    with patch("ml_switcheroo_compiler.ops.default_backend") as mock_op:
        mod.default_backend()
        mock_op.assert_called_once_with()


def test_device_count() -> None:
    """Test device_count."""
    with patch("ml_switcheroo_compiler.ops.device_count") as mock_op:
        mod.device_count()
        mock_op.assert_called_once_with()


def test_devices() -> None:
    """Test devices."""
    with patch("ml_switcheroo_compiler.ops.devices") as mock_op:
        mod.devices()
        mock_op.assert_called_once_with()


def test_discover_pjrt_plugins() -> None:
    """Test discover_pjrt_plugins."""
    with patch("ml_switcheroo_compiler.ops.discover_pjrt_plugins") as mock_op:
        mod.discover_pjrt_plugins()
        mock_op.assert_called_once_with()


def test_expand_platform_alias() -> None:
    """Test expand_platform_alias."""
    with patch("ml_switcheroo_compiler.ops.expand_platform_alias") as mock_op:
        mod.expand_platform_alias()
        mock_op.assert_called_once_with()


def test_get_backend() -> None:
    """Test get_backend."""
    with patch("ml_switcheroo_compiler.ops.get_backend") as mock_op:
        mod.get_backend()
        mock_op.assert_called_once_with()


def test_get_device_backend() -> None:
    """Test get_device_backend."""
    with patch("ml_switcheroo_compiler.ops.get_device_backend") as mock_op:
        mod.get_device_backend()
        mock_op.assert_called_once_with()


def test_get_tpu_library_path() -> None:
    """Test get_tpu_library_path."""
    with patch("ml_switcheroo_compiler.ops.get_tpu_library_path") as mock_op:
        mod.get_tpu_library_path()
        mock_op.assert_called_once_with()


def test_host_count() -> None:
    """Test host_count."""
    with patch("ml_switcheroo_compiler.ops.host_count") as mock_op:
        mod.host_count()
        mock_op.assert_called_once_with()


def test_host_id() -> None:
    """Test host_id."""
    with patch("ml_switcheroo_compiler.ops.host_id") as mock_op:
        mod.host_id()
        mock_op.assert_called_once_with()


def test_host_ids() -> None:
    """Test host_ids."""
    with patch("ml_switcheroo_compiler.ops.host_ids") as mock_op:
        mod.host_ids()
        mock_op.assert_called_once_with()


def test_is_gpu() -> None:
    """Test is_gpu."""
    with patch("ml_switcheroo_compiler.ops.is_gpu") as mock_op:
        mod.is_gpu()
        mock_op.assert_called_once_with()


def test_is_known_platform() -> None:
    """Test is_known_platform."""
    with patch("ml_switcheroo_compiler.ops.is_known_platform") as mock_op:
        mod.is_known_platform()
        mock_op.assert_called_once_with()


def test_local_device_count() -> None:
    """Test local_device_count."""
    with patch("ml_switcheroo_compiler.ops.local_device_count") as mock_op:
        mod.local_device_count()
        mock_op.assert_called_once_with()


def test_local_devices() -> None:
    """Test local_devices."""
    with patch("ml_switcheroo_compiler.ops.local_devices") as mock_op:
        mod.local_devices()
        mock_op.assert_called_once_with()


def test_lru_cache() -> None:
    """Test lru_cache."""
    with patch("ml_switcheroo_compiler.ops.lru_cache") as mock_op:
        mod.lru_cache()
        mock_op.assert_called_once_with()


def test_make_cpu_client() -> None:
    """Test make_cpu_client."""
    with patch("ml_switcheroo_compiler.ops.make_cpu_client") as mock_op:
        mod.make_cpu_client()
        mock_op.assert_called_once_with()


def test_make_gpu_client() -> None:
    """Test make_gpu_client."""
    with patch("ml_switcheroo_compiler.ops.make_gpu_client") as mock_op:
        mod.make_gpu_client()
        mock_op.assert_called_once_with()


def test_make_pjrt_topology() -> None:
    """Test make_pjrt_topology."""
    with patch("ml_switcheroo_compiler.ops.make_pjrt_topology") as mock_op:
        mod.make_pjrt_topology()
        mock_op.assert_called_once_with()


def test_make_pjrt_tpu_topology() -> None:
    """Test make_pjrt_tpu_topology."""
    with patch("ml_switcheroo_compiler.ops.make_pjrt_tpu_topology") as mock_op:
        mod.make_pjrt_tpu_topology()
        mock_op.assert_called_once_with()


def test_partial() -> None:
    """Test partial."""
    obj = mod.partial()
    assert obj is not None


def test_process_count() -> None:
    """Test process_count."""
    with patch("ml_switcheroo_compiler.ops.process_count") as mock_op:
        mod.process_count()
        mock_op.assert_called_once_with()


def test_process_index() -> None:
    """Test process_index."""
    with patch("ml_switcheroo_compiler.ops.process_index") as mock_op:
        mod.process_index()
        mock_op.assert_called_once_with()


def test_register_backend_factory() -> None:
    """Test register_backend_factory."""
    with patch("ml_switcheroo_compiler.ops.register_backend_factory") as mock_op:
        mod.register_backend_factory()
        mock_op.assert_called_once_with()


def test_register_pjrt_plugin_factories_from_env() -> None:
    """Test register_pjrt_plugin_factories_from_env."""
    with patch(
        "ml_switcheroo_compiler.ops.register_pjrt_plugin_factories_from_env"
    ) as mock_op:
        mod.register_pjrt_plugin_factories_from_env()
        mock_op.assert_called_once_with()


def test_register_plugin() -> None:
    """Test register_plugin."""
    with patch("ml_switcheroo_compiler.ops.register_plugin") as mock_op:
        mod.register_plugin()
        mock_op.assert_called_once_with()


def test_register_plugin_callbacks() -> None:
    """Test register_plugin_callbacks."""
    with patch("ml_switcheroo_compiler.ops.register_plugin_callbacks") as mock_op:
        mod.register_plugin_callbacks()
        mock_op.assert_called_once_with()


def test_tpu_client_timer_callback() -> None:
    """Test tpu_client_timer_callback."""
    with patch("ml_switcheroo_compiler.ops.tpu_client_timer_callback") as mock_op:
        mod.tpu_client_timer_callback()
        mock_op.assert_called_once_with()


def test_using_pjrt_c_api() -> None:
    """Test using_pjrt_c_api."""
    with patch("ml_switcheroo_compiler.ops.using_pjrt_c_api") as mock_op:
        mod.using_pjrt_c_api()
        mock_op.assert_called_once_with()
