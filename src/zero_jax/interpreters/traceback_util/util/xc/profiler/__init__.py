"""Frontend API routing for zero-jax.interpreters.traceback_util.util.xc.profiler."""

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class TraceMe:
    def __init__(self, *args, **kwargs):
        pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        def stub(*args, **kwargs):
            raise NotImplementedError(f"Stub for {name} is not implemented in backend")

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
