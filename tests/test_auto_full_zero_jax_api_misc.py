"""Tests for zero_jax.api.misc."""

from __future__ import annotations

import numpy as np

import zero_jax
import zero_jax.api.misc as mod
import zero_jax.numpy as jnp
from zero_jax._types import PyTree


def test_checkpoint_and_remat() -> None:
    """Verify checkpoint and remat both direct and parameterized."""

    @mod.checkpoint
    def f1(x: PyTree) -> PyTree:
        """Direct checkpointed function."""
        return x + 1

    @mod.checkpoint(prevent_cse=False, policy=mod.checkpoint_policies.everything_saveable)
    def f2(x: PyTree) -> PyTree:
        """Parameterized checkpointed function."""
        return x * 2

    assert f1(3) == 4
    assert f2(3) == 6
    assert mod.remat(lambda x: x - 1)(5) == 4


def test_closure_convert_and_make_jaxpr() -> None:
    """Verify closure_convert and make_jaxpr."""

    class CallableObj:
        """Callable object holding an ndarray attribute."""

        def __init__(self) -> None:
            """Initialize CallableObj."""
            self.w = jnp.array([1.0, 2.0])

        def __call__(self, x: PyTree) -> PyTree:
            """Apply CallableObj."""
            return x + self.w

    obj = CallableObj()
    conv_fn, aux = mod.closure_convert(obj, jnp.array([3.0, 4.0]))
    assert len(aux) == 1
    np.testing.assert_allclose(np.asarray(conv_fn(jnp.array([3.0, 4.0]))), np.array([4.0, 6.0]))

    # Builtin callable without __dict__
    assert mod.getattr_safe_dict(len) == {}

    jaxpr_repr = mod.make_jaxpr(lambda x: x + 1)(jnp.array([1.0, 2.0]))
    assert "lambda" in repr(jaxpr_repr)

    jaxpr_repr2, out_shape = mod.make_jaxpr(lambda x: x + 1, return_shape=True)(jnp.array([1.0, 2.0]))
    assert "lambda" in repr(jaxpr_repr2)
    assert out_shape.shape == (2,)


def test_ensure_compile_time_eval_and_misc_helpers() -> None:
    """Verify ensure_compile_time_eval and miscellaneous helpers."""
    fn = mod.ensure_compile_time_eval(lambda x: x * 3)
    assert callable(fn) and fn(4) == 12
    import contextlib

    ctx = mod.ensure_compile_time_eval()
    assert isinstance(ctx, contextlib.AbstractContextManager)
    with ctx:
        pass

    assert mod.named_call(lambda x: x + 2, name="add2")(3) == 5
    with mod.named_scope("my_scope"):
        pass

    assert mod.pure_callback(lambda x: x * 4, None, 3) == 12
    assert mod.make_array_from_callback((2, 2), None, lambda idx: 0).shape == (2, 2)
    assert mod.make_array_from_process_local_data(None, 42) == 42
    assert mod.make_array_from_single_device_arrays((2,), None, [jnp.array(1.0), jnp.array(2.0)]).shape == (2,)
    assert mod.softmax_custom_jvp(jnp.array([1.0, 2.0])).shape == (2,)

    mod.enable_custom_vjp_by_custom_transpose(True)
    mod.enable_custom_prng(True)
    mod.jax2tf_associative_scan_reductions()
    assert mod.default_prng_impl() == "threefry2x32"
    assert mod.default_matmul_precision() == "highest"
    assert mod.float0.name == "float0"

    with mod.threefry_partitionable():
        pass
    with mod.legacy_prng_key():
        pass
    with mod.debug_key_reuse():
        pass

    # Exercise jit stage methods and disable_jit
    j_fn = zero_jax.jit(lambda x: x + 1)
    assert j_fn.trace(jnp.array(1.0)) is not None
    assert j_fn.lower(jnp.array(1.0)) is not None
    with zero_jax.disable_jit():
        assert j_fn(jnp.array(2.0)) == jnp.array(3.0)

    j_static = zero_jax.jit(lambda x, n: x + n, static_argnums=1)
    assert j_static(jnp.array(2.0), 5) == jnp.array(7.0)
