"""Transformations for zero_jax."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import TensorConfig
from typing import Callable, Any
import contextlib
import functools


def jit(fun: Callable) -> Callable:
    """Compiles a function to execute faster.

    Args:
        fun: The function to be JIT-compiled.

    Returns:
        A wrapped version of the input function.
    """
    cache = {}

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """JAX API implementation for wrapped.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: The result.
        """
        from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
        from ml_switcheroo_ir import LogicalNode
        from ml_switcheroo_compiler.interpreter import evaluate_graph
        from zero_jax.numpy.lax_numpy import _to_tensor, ndarray, array
        import uuid

        # Simple cache key based on shapes and dtypes
        t_args = [_to_tensor(a) for a in args]
        key = tuple((t.shape, t.dtype.value) for t in t_args)

        if key not in cache:
            prev_graph = _tracer.active_graph
            is_tracing = _tracer.is_tracing
            graph = _tracer.start_tracing(name=f"jit_{fun.__name__}")

            proxy_args = []
            input_ids = []
            for arg in t_args:
                in_id = str(uuid.uuid4())
                input_ids.append(in_id)
                node = LogicalNode(
                    id=in_id, op_type="Input", inputs=[], shape_metadata=arg.shape
                )
                graph.nodes[in_id] = node
                proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
                from ml_switcheroo_compiler import Tensor

                proxy_tensor = Tensor(
                    data=proxy,
                    config=TensorConfig(
                        shape=arg.shape, dtype=arg.dtype, device=arg.device
                    ),
                )
                proxy_args.append(ndarray(proxy_tensor))

            out = fun(*proxy_args, **kwargs)
            out_tensor = _to_tensor(out)
            if out_tensor.data.id not in graph.outputs:
                graph.outputs.append(out_tensor.data.id)

            _tracer.stop_tracing()
            _tracer.active_graph = prev_graph
            _tracer.is_tracing = is_tracing

            cache[key] = (graph, input_ids, out_tensor.data.id)

        graph, input_ids, out_id = cache[key]

        from zero_jax.numpy import tensor_utils

        inputs = {
            in_id: tensor_utils.to_array(a.data) for in_id, a in zip(input_ids, t_args)
        }
        res = evaluate_graph(graph, inputs)

        return array(res[out_id])

    return wrapped


def grad(fun: Callable, argnums: Any = 0) -> Callable:
    """Creates a function that evaluates the gradient of fun.

    Args:
        fun: The function to be differentiated.
        argnums: Specifies which positional argument(s) to differentiate with respect to.

    Returns:
        A function that evaluates the gradient of the original function.
    """

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Executes the function and calculates its gradient.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            An array representing the computed gradient.
        """
        from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
        from ml_switcheroo_ir import LogicalNode
        from ml_switcheroo_compiler.transforms.autodiff import grad as ir_grad
        from ml_switcheroo_compiler.interpreter import evaluate_graph
        from zero_jax.numpy.lax_numpy import _to_tensor, array
        import uuid
        from zero_jax.numpy import tensor_utils

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
            from ml_switcheroo_compiler import Tensor

            proxy_tensor = Tensor(
                data=proxy,
                config=TensorConfig(
                    shape=arg.shape, dtype=arg.dtype, device=arg.device
                ),
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
    """Creates a function that evaluates both the value and gradient of fun.

    Args:
        fun: The function to be differentiated.
        argnums: Specifies which positional argument(s) to differentiate with respect to.

    Returns:
        A function that evaluates both the original function's value and its gradient.
    """

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Executes the function and calculates its value and gradient.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A tuple containing the evaluated value and its gradient.
        """
        val = fun(*args, **kwargs)
        g = grad(fun, argnums=argnums)(*args, **kwargs)
        return val, g

    return wrapped


def vmap(fun: Callable) -> Callable:
    """Vectorizing map. Creates a function which maps fun over argument axes.

    Args:
        fun: The function to be mapped over argument axes.

    Returns:
        A vectorized version of the input function.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Executes the vectorized function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the mapped function.
        """
        t_args = [
            a if hasattr(a, "__call__") or hasattr(a, "state") else _to_tensor(a)
            for a in args
        ]

        def inner_fun(*inner_args: Any) -> Any:
            """Executes the mapped function over mapped arguments.

            Args:
                *inner_args: Positional mapped arguments to pass to the function.

            Returns:
                The return value of the underlying function.
            """
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
    """A context manager to temporarily disable JIT compilation.

    Args:
        disable: Boolean to decide whether to disable JIT. Defaults to True.

    Yields:
        None, it simply provides a context where JIT compilation is disabled.
    """
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
    """Parallel map. Creates a function which evaluates fun in parallel on multiple XLA devices.

    Args:
        fun: The function to be mapped in parallel.
        axis_name: The name of the mapped axis.
        in_axes: Specifies the axes of the inputs to be mapped over.
        out_axes: Specifies where the mapped axis should appear in the output.
        static_broadcasted_argnums: Arguments to be treated as static (not mapped over).
        devices: Devices to use for parallel execution.
        backend: Backend to use for parallel execution.
        axis_size: The size of the mapped axis.
        donate_argnums: Arguments whose buffers can be donated to the computation.
        global_arg_shapes: Shapes of global arguments.

    Returns:
        A parallelized version of the input function.
    """
    return vmap(fun)


def eval_shape(fun: Callable, *args: Any, **kwargs: Any) -> Any:
    """Evaluates the shape and dtype of the output of fun without computing its values.

    Args:
        fun: The function whose output shape is evaluated.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        An object (or tree of objects) representing the shape and dtype of the output.
    """
    # A dummy eval_shape that just executes with Eager mode to get the shape wrapper
    from zero_jax.numpy.lax_numpy import _to_tensor
    import ml_switcheroo_compiler

    # Actually, proper eval_shape would trace without executing, but since eager mode returns
    # zeros of correct shape during tracing... Wait, if we use Tracer:
    from ml_switcheroo_compiler.tracing import _tracer

    # For now, just execute it and return the result which has a .shape
    # If we want pure shape, we can run it.
    with ml_switcheroo_compiler.EagerMode():
        res = fun(*args, **kwargs)

    class ShapedArray:
        """A simple wrapper containing shape and dtype information.

        Attributes:
            shape: The shape of the array.
            dtype: The dtype of the array.
        """

        def __init__(self, shape: Any, dtype: Any) -> None:
            """Initializes ShapedArray.

            Args:
                shape: The shape of the array.
                dtype: The dtype of the array.
            """
            self.shape = shape
            self.dtype = dtype

    t = _to_tensor(res)
    return ShapedArray(t.shape, t.dtype)
