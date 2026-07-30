import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import zero_jax
from zero_jax.numpy import array


def test_device_properties():
    assert zero_jax.default_backend() == "cpu"
    assert zero_jax.device_count() == 1
    assert zero_jax.local_device_count() == 1
    assert zero_jax.process_count() == 1
    assert zero_jax.process_index() == 0
    assert zero_jax.host_count() == 1
    assert zero_jax.host_id() == 0
    assert zero_jax.host_ids() == [0]


def test_types():
    assert isinstance(zero_jax.ShapeDtypeStruct((2, 2), int), zero_jax.ShapeDtypeStruct)
    assert isinstance(zero_jax.Shard("cpu", 0), zero_jax.Shard)
    assert isinstance(zero_jax.NamedSharding(None, None), zero_jax.NamedSharding)


def test_device_put():
    assert zero_jax.device_put(5) == 5
    assert zero_jax.device_put_replicated(5, [1, 2]) == [5, 5]


def test_debug_leaks():
    assert not zero_jax.checking_leaks()
    with zero_jax.check_tracer_leaks():
        pass
    zero_jax.clear_caches()
    with zero_jax.debug_infs():
        pass
    with zero_jax.debug_nans():
        pass


def test_misc():
    zero_jax.enable_checks()
    zero_jax.print_environment_info()
    zero_jax.effects_barrier()
    assert zero_jax.live_arrays() == []

    with zero_jax.log_compiles():
        pass
    with zero_jax.numpy_dtype_promotion("standard"):
        pass
    with zero_jax.numpy_rank_promotion("standard"):
        pass
    with zero_jax.spmd_mode("standard"):
        pass

    with zero_jax.transfer_guard("allow"):
        pass
    with zero_jax.transfer_guard_device_to_device("allow"):
        pass
    with zero_jax.transfer_guard_device_to_host("allow"):
        pass
    with zero_jax.transfer_guard_host_to_device("allow"):
        pass


def test_ad():
    import numpy as np

    def f(x):
        return x * x

    np.testing.assert_allclose(zero_jax.jacfwd(f)(array(1.0)), 2.0)
    np.testing.assert_allclose(zero_jax.jacrev(f)(array(1.0)), 2.0)
    np.testing.assert_allclose(zero_jax.jacobian(f)(array(1.0)), 2.0)

    val, _ = zero_jax.jvp(f, (array(1.0),), (array(1.0),))
    np.testing.assert_allclose(val, 1.0)
    val, _ = zero_jax.vjp(f, array(1.0))
    np.testing.assert_allclose(val, 1.0)
    val, _ = zero_jax.linearize(f, array(1.0))
    np.testing.assert_allclose(val, 1.0)


def test_more_misc():
    def f(x):
        return x

    f2, _ = zero_jax.closure_convert(f)
    assert f2(1) == 1
    assert zero_jax.named_call(f, "name")(1) == 1
    with zero_jax.named_scope("scope"):
        pass

    assert zero_jax.remat(f)(1) == 1
    assert zero_jax.checkpoint(f)(1) == 1
    assert zero_jax.ensure_compile_time_eval(f)(1) == 1

    assert zero_jax.pure_callback(f, None, 5) == 5

    assert zero_jax.make_array_from_process_local_data(None, 5) == 5

    zero_jax.enable_custom_vjp_by_custom_transpose()
    assert zero_jax.default_prng_impl() == "threefry2x32"
    zero_jax.jax2tf_associative_scan_reductions()
    assert zero_jax.default_matmul_precision() == "highest"
    with zero_jax.debug_key_reuse():
        pass


def test_even_more_misc():
    def f(x):
        return x

    f2 = zero_jax.custom_gradient(f)
    assert f2(1) == 1

    assert zero_jax.linear_transpose(f)(1) == 1
    assert isinstance(zero_jax.float0, type(zero_jax.float0))
    assert isinstance(
        zero_jax.threefry_partitionable, type(zero_jax.threefry_partitionable)
    )
    assert isinstance(zero_jax.checkpoint_policies, type(zero_jax.checkpoint_policies))
    assert isinstance(zero_jax.legacy_prng_key, type(zero_jax.legacy_prng_key))
    zero_jax.enable_custom_prng()

    f3 = zero_jax.custom_jvp(f)
    f3.defjvp(lambda x: None)

    f4 = zero_jax.custom_vjp(f)
    f4.defvjp(lambda x: None, lambda x: None)

    assert zero_jax.softmax_custom_jvp(array([1.0, 2.0])).shape == (2,)

    zero_jax.make_jaxpr(f)(1)

    assert zero_jax.make_array_from_single_device_arrays(
        (1,), None, [array(1.0)]
    ).shape == (1,)
    assert zero_jax.make_array_from_callback((2,), None, lambda: None).shape == (2,)
    assert zero_jax.device_put_sharded([array(1.0)], ["cpu"]).shape == (1,)
    assert zero_jax.block_until_ready(5) == 5


def test_phase5_stubs():
    import zero_jax.experimental.x64_context.config.profiler as c2
    import zero_jax.interpreters.partial_eval as pe
    import zero_jax.interpreters.partial_eval.config as pec
    import zero_jax.interpreters.traceback_util.config as tuc
    import zero_jax.interpreters.traceback_util.util.jaxlib_utils as tuju

    assert hasattr(pe, "JaxprTrace")
    assert hasattr(pec, "jax_jit")
    assert hasattr(tuc, "xla_client")
