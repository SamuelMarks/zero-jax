"""Tests for zero_jax module."""

import pytest
import zero_jax.lib.xla_client as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None


def test_CurrentSourceInfoMetadata():
    with pytest.raises(NotImplementedError):
        mod.CurrentSourceInfoMetadata()


def test_LoadedExecutable_execute():
    with pytest.raises(NotImplementedError):
        mod.LoadedExecutable_execute()


def test_LoadedExecutable_execute_with_token():
    with pytest.raises(NotImplementedError):
        mod.LoadedExecutable_execute_with_token()


def test_dtype_to_etype():
    with pytest.raises(NotImplementedError):
        mod.dtype_to_etype()


def test_execute_with_python_values():
    with pytest.raises(NotImplementedError):
        mod.execute_with_python_values()


def test_execute_with_python_values_replicated():
    with pytest.raises(NotImplementedError):
        mod.execute_with_python_values_replicated()


def test_generate_pjrt_gpu_plugin_options():
    with pytest.raises(NotImplementedError):
        mod.generate_pjrt_gpu_plugin_options()


def test_heap_profile():
    with pytest.raises(NotImplementedError):
        mod.heap_profile()


def test_initialize_pjrt_plugin():
    with pytest.raises(NotImplementedError):
        mod.initialize_pjrt_plugin()


def test_load_pjrt_plugin_dynamically():
    with pytest.raises(NotImplementedError):
        mod.load_pjrt_plugin_dynamically()


def test_load_pjrt_plugin_with_c_api():
    with pytest.raises(NotImplementedError):
        mod.load_pjrt_plugin_with_c_api()


def test_make_c_api_client():
    with pytest.raises(NotImplementedError):
        mod.make_c_api_client()


def test_make_c_api_device_topology():
    with pytest.raises(NotImplementedError):
        mod.make_c_api_device_topology()


def test_make_convolution_dimension_numbers():
    with pytest.raises(NotImplementedError):
        mod.make_convolution_dimension_numbers()


def test_make_cpu_client():
    with pytest.raises(NotImplementedError):
        mod.make_cpu_client()


def test_make_dot_dimension_numbers():
    with pytest.raises(NotImplementedError):
        mod.make_dot_dimension_numbers()


def test_make_gpu_client():
    with pytest.raises(NotImplementedError):
        mod.make_gpu_client()


def test_make_padding_config():
    with pytest.raises(NotImplementedError):
        mod.make_padding_config()


def test_make_replica_groups():
    with pytest.raises(NotImplementedError):
        mod.make_replica_groups()


def test_make_tfrt_tpu_c_api_client():
    with pytest.raises(NotImplementedError):
        mod.make_tfrt_tpu_c_api_client()


def test_make_tfrt_tpu_c_api_device_topology():
    with pytest.raises(NotImplementedError):
        mod.make_tfrt_tpu_c_api_device_topology()


def test_make_tpu_client():
    with pytest.raises(NotImplementedError):
        mod.make_tpu_client()


def test_pjrt_plugin_initialized():
    with pytest.raises(NotImplementedError):
        mod.pjrt_plugin_initialized()


def test_pjrt_plugin_loaded():
    with pytest.raises(NotImplementedError):
        mod.pjrt_plugin_loaded()


def test_register_custom_call_handler():
    with pytest.raises(NotImplementedError):
        mod.register_custom_call_handler()


def test_register_custom_call_target():
    with pytest.raises(NotImplementedError):
        mod.register_custom_call_target()


def test_shape_from_pyval():
    with pytest.raises(NotImplementedError):
        mod.shape_from_pyval()


def test_tracebacks():
    with pytest.raises(NotImplementedError):
        mod.tracebacks()


def test_window_padding_type_to_pad_values():
    with pytest.raises(NotImplementedError):
        mod.window_padding_type_to_pad_values()
