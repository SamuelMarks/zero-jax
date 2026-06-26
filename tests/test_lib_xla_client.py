"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.lib.xla_client as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None


def test_CurrentSourceInfoMetadata():
    with patch("ml_switcheroo_compiler.ops.CurrentSourceInfoMetadata") as mock_op:
        mod.CurrentSourceInfoMetadata()
        mock_op.assert_called_once_with()


def test_LoadedExecutable_execute():
    with patch("ml_switcheroo_compiler.ops.LoadedExecutable_execute") as mock_op:
        mod.LoadedExecutable_execute()
        mock_op.assert_called_once_with()


def test_LoadedExecutable_execute_with_token():
    with patch(
        "ml_switcheroo_compiler.ops.LoadedExecutable_execute_with_token"
    ) as mock_op:
        mod.LoadedExecutable_execute_with_token()
        mock_op.assert_called_once_with()


def test_dtype_to_etype():
    with patch("ml_switcheroo_compiler.ops.dtype_to_etype") as mock_op:
        mod.dtype_to_etype()
        mock_op.assert_called_once_with()


def test_execute_with_python_values():
    with patch("ml_switcheroo_compiler.ops.execute_with_python_values") as mock_op:
        mod.execute_with_python_values()
        mock_op.assert_called_once_with()


def test_execute_with_python_values_replicated():
    with patch(
        "ml_switcheroo_compiler.ops.execute_with_python_values_replicated"
    ) as mock_op:
        mod.execute_with_python_values_replicated()
        mock_op.assert_called_once_with()


def test_generate_pjrt_gpu_plugin_options():
    with patch(
        "ml_switcheroo_compiler.ops.generate_pjrt_gpu_plugin_options"
    ) as mock_op:
        mod.generate_pjrt_gpu_plugin_options()
        mock_op.assert_called_once_with()


def test_heap_profile():
    with patch("ml_switcheroo_compiler.ops.heap_profile") as mock_op:
        mod.heap_profile()
        mock_op.assert_called_once_with()


def test_initialize_pjrt_plugin():
    with patch("ml_switcheroo_compiler.ops.initialize_pjrt_plugin") as mock_op:
        mod.initialize_pjrt_plugin()
        mock_op.assert_called_once_with()


def test_load_pjrt_plugin_dynamically():
    with patch("ml_switcheroo_compiler.ops.load_pjrt_plugin_dynamically") as mock_op:
        mod.load_pjrt_plugin_dynamically()
        mock_op.assert_called_once_with()


def test_load_pjrt_plugin_with_c_api():
    with patch("ml_switcheroo_compiler.ops.load_pjrt_plugin_with_c_api") as mock_op:
        mod.load_pjrt_plugin_with_c_api()
        mock_op.assert_called_once_with()


def test_make_c_api_client():
    with patch("ml_switcheroo_compiler.ops.make_c_api_client") as mock_op:
        mod.make_c_api_client()
        mock_op.assert_called_once_with()


def test_make_c_api_device_topology():
    with patch("ml_switcheroo_compiler.ops.make_c_api_device_topology") as mock_op:
        mod.make_c_api_device_topology()
        mock_op.assert_called_once_with()


def test_make_convolution_dimension_numbers():
    with patch(
        "ml_switcheroo_compiler.ops.make_convolution_dimension_numbers"
    ) as mock_op:
        mod.make_convolution_dimension_numbers()
        mock_op.assert_called_once_with()


def test_make_cpu_client():
    with patch("ml_switcheroo_compiler.ops.make_cpu_client") as mock_op:
        mod.make_cpu_client()
        mock_op.assert_called_once_with()


def test_make_dot_dimension_numbers():
    with patch("ml_switcheroo_compiler.ops.make_dot_dimension_numbers") as mock_op:
        mod.make_dot_dimension_numbers()
        mock_op.assert_called_once_with()


def test_make_gpu_client():
    with patch("ml_switcheroo_compiler.ops.make_gpu_client") as mock_op:
        mod.make_gpu_client()
        mock_op.assert_called_once_with()


def test_make_padding_config():
    with patch("ml_switcheroo_compiler.ops.make_padding_config") as mock_op:
        mod.make_padding_config()
        mock_op.assert_called_once_with()


def test_make_replica_groups():
    with patch("ml_switcheroo_compiler.ops.make_replica_groups") as mock_op:
        mod.make_replica_groups()
        mock_op.assert_called_once_with()


def test_make_tfrt_tpu_c_api_client():
    with patch("ml_switcheroo_compiler.ops.make_tfrt_tpu_c_api_client") as mock_op:
        mod.make_tfrt_tpu_c_api_client()
        mock_op.assert_called_once_with()


def test_make_tfrt_tpu_c_api_device_topology():
    with patch(
        "ml_switcheroo_compiler.ops.make_tfrt_tpu_c_api_device_topology"
    ) as mock_op:
        mod.make_tfrt_tpu_c_api_device_topology()
        mock_op.assert_called_once_with()


def test_make_tpu_client():
    with patch("ml_switcheroo_compiler.ops.make_tpu_client") as mock_op:
        mod.make_tpu_client()
        mock_op.assert_called_once_with()


def test_pjrt_plugin_initialized():
    with patch("ml_switcheroo_compiler.ops.pjrt_plugin_initialized") as mock_op:
        mod.pjrt_plugin_initialized()
        mock_op.assert_called_once_with()


def test_pjrt_plugin_loaded():
    with patch("ml_switcheroo_compiler.ops.pjrt_plugin_loaded") as mock_op:
        mod.pjrt_plugin_loaded()
        mock_op.assert_called_once_with()


def test_register_custom_call_handler():
    with patch("ml_switcheroo_compiler.ops.register_custom_call_handler") as mock_op:
        mod.register_custom_call_handler()
        mock_op.assert_called_once_with()


def test_register_custom_call_target():
    with patch("ml_switcheroo_compiler.ops.register_custom_call_target") as mock_op:
        mod.register_custom_call_target()
        mock_op.assert_called_once_with()


def test_shape_from_pyval():
    with patch("ml_switcheroo_compiler.ops.shape_from_pyval") as mock_op:
        mod.shape_from_pyval()
        mock_op.assert_called_once_with()


def test_tracebacks():
    with patch("ml_switcheroo_compiler.ops.tracebacks") as mock_op:
        mod.tracebacks()
        mock_op.assert_called_once_with()


def test_window_padding_type_to_pad_values():
    with patch(
        "ml_switcheroo_compiler.ops.window_padding_type_to_pad_values"
    ) as mock_op:
        mod.window_padding_type_to_pad_values()
        mock_op.assert_called_once_with()
