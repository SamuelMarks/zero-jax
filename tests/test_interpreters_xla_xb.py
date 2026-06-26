"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb as mod


def test_BackendFactory() -> None:
    """Test BackendFactory."""
    with pytest.raises(NotImplementedError):
        mod.BackendFactory()


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
    with pytest.raises(NotImplementedError):
        mod.TopologyFactory()


def test_XlaBackend() -> None:
    """Test XlaBackend."""
    obj = mod.XlaBackend()
    assert obj is not None


def test_backend_pjrt_c_api_version() -> None:
    """Test backend_pjrt_c_api_version."""
    with pytest.raises(NotImplementedError):
        mod.backend_pjrt_c_api_version()


def test_backend_xla_version() -> None:
    """Test backend_xla_version."""
    with pytest.raises(NotImplementedError):
        mod.backend_xla_version()


def test_backends() -> None:
    """Test backends."""
    with pytest.raises(NotImplementedError):
        mod.backends()


def test_backends_are_initialized() -> None:
    """Test backends_are_initialized."""
    with pytest.raises(NotImplementedError):
        mod.backends_are_initialized()


def test_canonicalize_platform() -> None:
    """Test canonicalize_platform."""
    with pytest.raises(NotImplementedError):
        mod.canonicalize_platform()


def test_default_backend() -> None:
    """Test default_backend."""
    with pytest.raises(NotImplementedError):
        mod.default_backend()


def test_device_count() -> None:
    """Test device_count."""
    with pytest.raises(NotImplementedError):
        mod.device_count()


def test_devices() -> None:
    """Test devices."""
    with pytest.raises(NotImplementedError):
        mod.devices()


def test_discover_pjrt_plugins() -> None:
    """Test discover_pjrt_plugins."""
    with pytest.raises(NotImplementedError):
        mod.discover_pjrt_plugins()


def test_expand_platform_alias() -> None:
    """Test expand_platform_alias."""
    with pytest.raises(NotImplementedError):
        mod.expand_platform_alias()


def test_get_backend() -> None:
    """Test get_backend."""
    with pytest.raises(NotImplementedError):
        mod.get_backend()


def test_get_device_backend() -> None:
    """Test get_device_backend."""
    with pytest.raises(NotImplementedError):
        mod.get_device_backend()


def test_get_tpu_library_path() -> None:
    """Test get_tpu_library_path."""
    with pytest.raises(NotImplementedError):
        mod.get_tpu_library_path()


def test_host_count() -> None:
    """Test host_count."""
    with pytest.raises(NotImplementedError):
        mod.host_count()


def test_host_id() -> None:
    """Test host_id."""
    with pytest.raises(NotImplementedError):
        mod.host_id()


def test_host_ids() -> None:
    """Test host_ids."""
    with pytest.raises(NotImplementedError):
        mod.host_ids()


def test_is_gpu() -> None:
    """Test is_gpu."""
    with pytest.raises(NotImplementedError):
        mod.is_gpu()


def test_is_known_platform() -> None:
    """Test is_known_platform."""
    with pytest.raises(NotImplementedError):
        mod.is_known_platform()


def test_local_device_count() -> None:
    """Test local_device_count."""
    with pytest.raises(NotImplementedError):
        mod.local_device_count()


def test_local_devices() -> None:
    """Test local_devices."""
    with pytest.raises(NotImplementedError):
        mod.local_devices()


def test_lru_cache() -> None:
    """Test lru_cache."""
    with pytest.raises(NotImplementedError):
        mod.lru_cache()


def test_make_cpu_client() -> None:
    """Test make_cpu_client."""
    with pytest.raises(NotImplementedError):
        mod.make_cpu_client()


def test_make_gpu_client() -> None:
    """Test make_gpu_client."""
    with pytest.raises(NotImplementedError):
        mod.make_gpu_client()


def test_make_pjrt_topology() -> None:
    """Test make_pjrt_topology."""
    with pytest.raises(NotImplementedError):
        mod.make_pjrt_topology()


def test_make_pjrt_tpu_topology() -> None:
    """Test make_pjrt_tpu_topology."""
    with pytest.raises(NotImplementedError):
        mod.make_pjrt_tpu_topology()


def test_partial() -> None:
    """Test partial."""
    obj = mod.partial()
    assert obj is not None


def test_process_count() -> None:
    """Test process_count."""
    with pytest.raises(NotImplementedError):
        mod.process_count()


def test_process_index() -> None:
    """Test process_index."""
    with pytest.raises(NotImplementedError):
        mod.process_index()


def test_register_backend_factory() -> None:
    """Test register_backend_factory."""
    with pytest.raises(NotImplementedError):
        mod.register_backend_factory()


def test_register_pjrt_plugin_factories_from_env() -> None:
    """Test register_pjrt_plugin_factories_from_env."""
    with pytest.raises(NotImplementedError):
        mod.register_pjrt_plugin_factories_from_env()


def test_register_plugin() -> None:
    """Test register_plugin."""
    with pytest.raises(NotImplementedError):
        mod.register_plugin()


def test_register_plugin_callbacks() -> None:
    """Test register_plugin_callbacks."""
    with pytest.raises(NotImplementedError):
        mod.register_plugin_callbacks()


def test_tpu_client_timer_callback() -> None:
    """Test tpu_client_timer_callback."""
    with pytest.raises(NotImplementedError):
        mod.tpu_client_timer_callback()


def test_using_pjrt_c_api() -> None:
    """Test using_pjrt_c_api."""
    with pytest.raises(NotImplementedError):
        mod.using_pjrt_c_api()
