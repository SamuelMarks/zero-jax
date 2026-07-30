"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.lib.xla_client as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None


def test_CurrentSourceInfoMetadata():
    with patch(
        "zero_jax._compiler_proxy_ops.CurrentSourceInfoMetadata", create=True
    ) as mock_op:
        mod.CurrentSourceInfoMetadata()
        mock_op.assert_called_once_with()


def test_LoadedExecutable_execute():
    with patch(
        "zero_jax._compiler_proxy_ops.LoadedExecutable_execute", create=True
    ) as mock_op:
        mod.LoadedExecutable_execute()
        mock_op.assert_called_once_with()


def test_LoadedExecutable_execute_with_token():
    with patch(
        "zero_jax._compiler_proxy_ops.LoadedExecutable_execute_with_token", create=True
    ) as mock_op:
        mod.LoadedExecutable_execute_with_token()
        mock_op.assert_called_once_with()


def test_dtype_to_etype():
    with patch("zero_jax._compiler_proxy_ops.dtype_to_etype", create=True) as mock_op:
        mod.dtype_to_etype()
        mock_op.assert_called_once_with()


def test_execute_with_python_values():
    with patch(
        "zero_jax._compiler_proxy_ops.execute_with_python_values", create=True
    ) as mock_op:
        mod.execute_with_python_values()
        mock_op.assert_called_once_with()


def test_execute_with_python_values_replicated():
    with patch(
        "zero_jax._compiler_proxy_ops.execute_with_python_values_replicated",
        create=True,
    ) as mock_op:
        mod.execute_with_python_values_replicated()
        mock_op.assert_called_once_with()


def test_generate_pjrt_gpu_plugin_options():
    with patch(
        "zero_jax._compiler_proxy_ops.generate_pjrt_gpu_plugin_options", create=True
    ) as mock_op:
        mod.generate_pjrt_gpu_plugin_options()
        mock_op.assert_called_once_with()


def test_heap_profile():
    with patch("zero_jax._compiler_proxy_ops.heap_profile", create=True) as mock_op:
        mod.heap_profile()
        mock_op.assert_called_once_with()


def test_initialize_pjrt_plugin():
    with patch(
        "zero_jax._compiler_proxy_ops.initialize_pjrt_plugin", create=True
    ) as mock_op:
        mod.initialize_pjrt_plugin()
        mock_op.assert_called_once_with()


def test_load_pjrt_plugin_dynamically():
    with patch(
        "zero_jax._compiler_proxy_ops.load_pjrt_plugin_dynamically", create=True
    ) as mock_op:
        mod.load_pjrt_plugin_dynamically()
        mock_op.assert_called_once_with()


def test_load_pjrt_plugin_with_c_api():
    with patch(
        "zero_jax._compiler_proxy_ops.load_pjrt_plugin_with_c_api", create=True
    ) as mock_op:
        mod.load_pjrt_plugin_with_c_api()
        mock_op.assert_called_once_with()


def test_make_c_api_client():
    with patch(
        "zero_jax._compiler_proxy_ops.make_c_api_client", create=True
    ) as mock_op:
        mod.make_c_api_client()
        mock_op.assert_called_once_with()


def test_make_c_api_device_topology():
    with patch(
        "zero_jax._compiler_proxy_ops.make_c_api_device_topology", create=True
    ) as mock_op:
        mod.make_c_api_device_topology()
        mock_op.assert_called_once_with()


def test_make_convolution_dimension_numbers():
    with patch(
        "zero_jax._compiler_proxy_ops.make_convolution_dimension_numbers", create=True
    ) as mock_op:
        mod.make_convolution_dimension_numbers()
        mock_op.assert_called_once_with()


def test_make_cpu_client():
    with patch("zero_jax._compiler_proxy_ops.make_cpu_client", create=True) as mock_op:
        mod.make_cpu_client()
        mock_op.assert_called_once_with()


def test_make_dot_dimension_numbers():
    with patch(
        "zero_jax._compiler_proxy_ops.make_dot_dimension_numbers", create=True
    ) as mock_op:
        mod.make_dot_dimension_numbers()
        mock_op.assert_called_once_with()


def test_make_gpu_client():
    with patch("zero_jax._compiler_proxy_ops.make_gpu_client", create=True) as mock_op:
        mod.make_gpu_client()
        mock_op.assert_called_once_with()


def test_make_padding_config():
    with patch(
        "zero_jax._compiler_proxy_ops.make_padding_config", create=True
    ) as mock_op:
        mod.make_padding_config()
        mock_op.assert_called_once_with()


def test_make_replica_groups():
    with patch(
        "zero_jax._compiler_proxy_ops.make_replica_groups", create=True
    ) as mock_op:
        mod.make_replica_groups()
        mock_op.assert_called_once_with()


def test_make_tfrt_tpu_c_api_client():
    with patch(
        "zero_jax._compiler_proxy_ops.make_tfrt_tpu_c_api_client", create=True
    ) as mock_op:
        mod.make_tfrt_tpu_c_api_client()
        mock_op.assert_called_once_with()


def test_make_tfrt_tpu_c_api_device_topology():
    with patch(
        "zero_jax._compiler_proxy_ops.make_tfrt_tpu_c_api_device_topology", create=True
    ) as mock_op:
        mod.make_tfrt_tpu_c_api_device_topology()
        mock_op.assert_called_once_with()


def test_make_tpu_client():
    with patch("zero_jax._compiler_proxy_ops.make_tpu_client", create=True) as mock_op:
        mod.make_tpu_client()
        mock_op.assert_called_once_with()


def test_pjrt_plugin_initialized():
    with patch(
        "zero_jax._compiler_proxy_ops.pjrt_plugin_initialized", create=True
    ) as mock_op:
        mod.pjrt_plugin_initialized()
        mock_op.assert_called_once_with()


def test_pjrt_plugin_loaded():
    with patch(
        "zero_jax._compiler_proxy_ops.pjrt_plugin_loaded", create=True
    ) as mock_op:
        mod.pjrt_plugin_loaded()
        mock_op.assert_called_once_with()


def test_register_custom_call_handler():
    with patch(
        "zero_jax._compiler_proxy_ops.register_custom_call_handler", create=True
    ) as mock_op:
        mod.register_custom_call_handler()
        mock_op.assert_called_once_with()


def test_register_custom_call_target():
    with patch(
        "zero_jax._compiler_proxy_ops.register_custom_call_target", create=True
    ) as mock_op:
        mod.register_custom_call_target()
        mock_op.assert_called_once_with()


def test_shape_from_pyval():
    with patch("zero_jax._compiler_proxy_ops.shape_from_pyval", create=True) as mock_op:
        mod.shape_from_pyval()
        mock_op.assert_called_once_with()


def test_tracebacks():
    with patch("zero_jax._compiler_proxy_ops.tracebacks", create=True) as mock_op:
        mod.tracebacks()
        mock_op.assert_called_once_with()


def test_window_padding_type_to_pad_values():
    with patch(
        "zero_jax._compiler_proxy_ops.window_padding_type_to_pad_values", create=True
    ) as mock_op:
        mod.window_padding_type_to_pad_values()
        mock_op.assert_called_once_with()
