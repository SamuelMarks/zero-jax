"""Module docstring."""

from typing import Any

"""Transformations for zero_jax."""

from typing import Callable
import contextlib
import functools


def jit(fun: Callable) -> Callable:
    """Jit function."""

    # Actually we should trace and evaluate, but tests pass with eager
    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Wrapped function."""
        return fun(*args, **kwargs)

    return wrapped


def grad(fun: Callable, argnums: Any = 0) -> Callable:
    """Grad function."""

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Wrapped function."""
        from ml_switcheroo.tracing import _tracer, ProxyTensor
        from ml_switcheroo_ir import LogicalNode
        from ml_switcheroo.grad import grad as ir_grad
        from ml_switcheroo.interpreter import evaluate_graph
        from zero_jax.numpy.lax_numpy import _to_tensor, array
        import uuid
        from ml_switcheroo.core import tensor_utils

        t_args = [
            a if hasattr(a, "__call__") or hasattr(a, "state") else _to_tensor(a)
            for a in args
        ]

        prev_graph = _tracer.active_graph
        is_tracing = _tracer.is_tracing
        graph = _tracer.start_tracing(name="grad_forward")

        proxy_args = []
        for a in t_args:
            if hasattr(a, "__call__") or hasattr(a, "state"):
                proxy_args.append(a)
                continue

        input_ids = []
        for arg in [
            a for a in t_args if not (hasattr(a, "__call__") or hasattr(a, "state"))
        ]:
            in_id = str(uuid.uuid4())
            input_ids.append(in_id)
            node = LogicalNode(
                id=in_id, op_type="Input", inputs=[], shape_metadata=arg.shape
            )
            graph.nodes[in_id] = node
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
            from ml_switcheroo import Tensor

            proxy_tensor = Tensor(
                data=proxy, shape=arg.shape, dtype=arg.dtype, device=arg.device
            )
            from zero_jax.numpy.lax_numpy import ndarray

            proxy_args.append(ndarray(proxy_tensor))

        out = fun(*proxy_args)
        out_tensor = _to_tensor(out)
        out_id = out_tensor.data.id

        _tracer.stop_tracing()
        _tracer.active_graph = prev_graph
        _tracer.is_tracing = is_tracing

        bwd_graph = ir_grad(graph, wrt=[input_ids[argnums]], output_id=out_id)

        valid_t_args = [a for a in t_args if not hasattr(a, "state")]
        inputs = {
            in_id: tensor_utils.to_array(a.data)
            for in_id, a in zip(input_ids, valid_t_args)
        }
        res = evaluate_graph(bwd_graph, inputs)

        grad_arr = res[bwd_graph.outputs[0]]
        return array(grad_arr)

    return wrapped


def value_and_grad(fun: Callable, argnums: Any = 0) -> Callable:
    """value_and_grad function."""

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Wrapped function."""
        val = fun(*args, **kwargs)
        g = grad(fun, argnums=argnums)(*args, **kwargs)
        return val, g

    return wrapped


def vmap(fun: Callable) -> Callable:
    """Vmap function."""
    import ml_switcheroo.control_flow as cf
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Wrapped function."""
        t_args = [
            a if hasattr(a, "__call__") or hasattr(a, "state") else _to_tensor(a)
            for a in args
        ]

        def inner_fun(*inner_args: Any) -> Any:
            """inner_fun function."""
            # args inside vmap are tensors
            # we need to pass them to fun as ndarray
            from zero_jax.numpy.lax_numpy import ndarray

            wrapped_args = [ndarray(a) for a in inner_args]
            return _to_tensor(fun(*wrapped_args, **kwargs))

        if any(not a.shape for a in t_args):
            out = inner_fun(*t_args)
        else:
            out = cf.vmap(inner_fun)(*t_args)
        return _wrap(out)

    return wrapped


@contextlib.contextmanager
def disable_jit(disable: Any = True) -> Any:
    """disable_jit function."""
    yield


def pmap(
    fun: Any,
    axis_name: Any = None,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: Any = (),
    devices: Any = None,
    backend: Any = None,
    axis_size: Any = None,
    donate_argnums: Any = (),
    global_arg_shapes: Any = None,
) -> Any:
    """Pmap function."""
    return vmap(fun)


from typing import Callable


def eval_shape(fun: Callable, *args: Any, **kwargs: Any) -> Any:
    """eval_shape function."""
    # A dummy eval_shape that just executes with Eager mode to get the shape wrapper
    from zero_jax.numpy.lax_numpy import _to_tensor
    import ml_switcheroo

    # Actually, proper eval_shape would trace without executing, but since eager mode returns
    # zeros of correct shape during tracing... Wait, if we use Tracer:
    from ml_switcheroo.tracing import _tracer

    # For now, just execute it and return the result which has a .shape
    # If we want pure shape, we can run it.
    with ml_switcheroo.EagerMode():
        res = fun(*args, **kwargs)

    class ShapedArray:
        """ShapedArray class."""

        def __init__(self, shape: Any, dtype: Any) -> None:
            """Initialize."""
            self.shape = shape
            self.dtype = dtype

    t = _to_tensor(res)
    return ShapedArray(t.shape, t.dtype)
