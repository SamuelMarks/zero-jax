"""Main initialization module for zero_jax."""

from __future__ import annotations

__version__ = "0.4.30"
__version_info__ = (0, 4, 30)

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler.core import EagerMode
from ml_switcheroo_compiler.core.config import config

import zero_jax._compiler_patches as _compiler_patches
import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue, T
from zero_jax.numpy.lax_numpy import ndarray as Array

_compiler_patches.register_compiler_patches()

from . import (
    ad_checkpoint,
    api_util,
    cloud_tpu_init,
    collect_profile,
    core,
    custom_batching,
    custom_derivatives,
    custom_transpose,
    debug,
    distributed,
    dlpack,
    dtypes,
    errors,
    example_libraries,
    experimental,
    export,
    extend,
    flatten_util,
    image,
    interpreters,
    lax,
    lib,
    monitoring,
    nn,
    ops,
    profiler,
    random,
    scipy,
    sharding,
    stages,
    test_util,
    tools,
    tree,
    tree_util,
    typing,
    util,
    version,
)
from ._config_internal import set_host_device_count
from .api import (
    NamedSharding,
    ShapeDtypeStruct,
    Shard,
    block_until_ready,
    check_tracer_leaks,
    checking_leaks,
    checkpoint,
    checkpoint_policies,
    clear_caches,
    closure_convert,
    custom_gradient,
    custom_jvp,
    custom_vjp,
    debug_infs,
    debug_key_reuse,
    debug_nans,
    default_backend,
    default_device,
    default_matmul_precision,
    default_prng_impl,
    device_count,
    device_put,
    device_put_replicated,
    device_put_sharded,
    disable_jit,
    effects_barrier,
    enable_checks,
    enable_custom_prng,
    enable_custom_vjp_by_custom_transpose,
    ensure_compile_time_eval,
    eval_shape,
    float0,
    grad,
    hessian,
    host_count,
    host_id,
    host_ids,
    jacfwd,
    jacobian,
    jacrev,
    jax2tf_associative_scan_reductions,
    jit,
    jvp,
    legacy_prng_key,
    linear_transpose,
    linearize,
    live_arrays,
    local_device_count,
    log_compiles,
    make_array_from_callback,
    make_array_from_process_local_data,
    make_array_from_single_device_arrays,
    make_jaxpr,
    named_call,
    named_scope,
    numpy_dtype_promotion,
    numpy_rank_promotion,
    pmap,
    print_environment_info,
    process_count,
    process_index,
    pure_callback,
    remat,
    softmax_custom_jvp,
    spmd_mode,
    threefry_partitionable,
    transfer_guard,
    transfer_guard_device_to_device,
    transfer_guard_device_to_host,
    transfer_guard_host_to_device,
    value_and_grad,
    vjp,
    vmap,
)
from .api.core import Device


def jax(version: str | None = None) -> None:
    """Stub callable for the top-level ``jax`` entry point.

    Args:
        version: Optional version query string.

    Returns:
        None: Always returns ``None``.
    """
    _ = version
    return


def devices(backend: str | None = None) -> list[Device]:
    """Return a list of all available devices for the given backend.

    Args:
        backend: Optional backend platform name.

    Returns:
        list[Device]: List of available devices.
    """
    b = backend or config.backend
    return [Device(platform=b, id=i) for i in range(device_count(b))]


def local_devices(backend: str | None = None) -> list[Device]:
    """Return a list of local devices addressable by the current process.

    Args:
        backend: Optional backend platform name.

    Returns:
        list[Device]: List of local devices.
    """
    b = backend or config.backend
    return [Device(platform=b, id=i) for i in range(local_device_count(b))]


def device_get(x: T) -> T:
    """Transfer a value or PyTree from device to host Python memory.

    Args:
        x: Input array or PyTree to retrieve.

    Returns:
        T: The host-resident value.
    """
    return x


config.eager_mode = True

__all__ = [
    "Array",
    "Device",
    "EagerMode",
    "NamedSharding",
    "ShapeDtypeStruct",
    "Shard",
    "_compiler_patches",
    "_ml_switcheroo_compiler",
    "_ops",
    "ad_checkpoint",
    "api_util",
    "block_until_ready",
    "check_tracer_leaks",
    "checking_leaks",
    "checkpoint",
    "checkpoint_policies",
    "clear_caches",
    "closure_convert",
    "cloud_tpu_init",
    "collect_profile",
    "core",
    "custom_batching",
    "custom_derivatives",
    "custom_gradient",
    "custom_jvp",
    "custom_transpose",
    "custom_vjp",
    "debug",
    "debug_infs",
    "debug_key_reuse",
    "debug_nans",
    "default_backend",
    "default_device",
    "default_matmul_precision",
    "default_prng_impl",
    "device_count",
    "device_get",
    "device_put",
    "device_put_replicated",
    "device_put_sharded",
    "devices",
    "disable_jit",
    "distributed",
    "dlpack",
    "dtypes",
    "effects_barrier",
    "enable_checks",
    "enable_custom_prng",
    "enable_custom_vjp_by_custom_transpose",
    "ensure_compile_time_eval",
    "errors",
    "eval_shape",
    "example_libraries",
    "experimental",
    "export",
    "extend",
    "flatten_util",
    "float0",
    "grad",
    "hessian",
    "host_count",
    "host_id",
    "host_ids",
    "image",
    "interpreters",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jax",
    "jax2tf_associative_scan_reductions",
    "jit",
    "jvp",
    "lax",
    "legacy_prng_key",
    "lib",
    "linear_transpose",
    "linearize",
    "live_arrays",
    "local_device_count",
    "local_devices",
    "log_compiles",
    "make_array_from_callback",
    "make_array_from_process_local_data",
    "make_array_from_single_device_arrays",
    "make_jaxpr",
    "monitoring",
    "named_call",
    "named_scope",
    "nn",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "ops",
    "pmap",
    "print_environment_info",
    "process_count",
    "process_index",
    "profiler",
    "pure_callback",
    "random",
    "remat",
    "scipy",
    "set_host_device_count",
    "sharding",
    "softmax_custom_jvp",
    "spmd_mode",
    "stages",
    "test_util",
    "threefry_partitionable",
    "tools",
    "transfer_guard",
    "transfer_guard_device_to_device",
    "transfer_guard_device_to_host",
    "transfer_guard_host_to_device",
    "tree",
    "tree_util",
    "typing",
    "util",
    "value_and_grad",
    "version",
    "vjp",
    "vmap",
]
