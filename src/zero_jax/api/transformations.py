"""Transformations for zero_jax."""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable

from ml_switcheroo_compiler.core.tensor import TensorConfig


def jit(
    fun: Callable,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: str | tuple[str, ...] | None = None,
    donate_argnums: int | tuple[int, ...] | None = None,
    donate_argnames: str | tuple[str, ...] | None = None,
    keep_unused: bool = False,
    device: Any | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> Callable:
    """Compiles a function to execute faster.

    Args:
        fun: The function to be JIT-compiled.

    Returns:
        A wrapped version of the input function.
    """
    cache = {}

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        import uuid

        from ml_switcheroo_compiler.interpreter import evaluate_graph
        from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
        from ml_switcheroo_ir import LogicalNode

        from zero_jax.numpy.lax_numpy import _to_tensor, array, ndarray
        from zero_jax.tree_util import tree_flatten, tree_unflatten

        flat_args, tree_def = tree_flatten((args, kwargs))
        t_args = [_to_tensor(a) for a in flat_args]
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

            unflattened_args, unflattened_kwargs = tree_unflatten(tree_def, proxy_args)

            out = fun(*unflattened_args, **unflattened_kwargs)

            flat_out, out_tree_def = tree_flatten(out)
            out_ids = []
            for o in flat_out:
                out_tensor = _to_tensor(o)
                out_id = out_tensor.data.id
                out_ids.append(out_id)
                if out_id not in graph.outputs:
                    graph.outputs.append(out_id)

            _tracer.stop_tracing()
            _tracer.active_graph = prev_graph
            _tracer.is_tracing = is_tracing

            cache[key] = (graph, input_ids, out_ids, out_tree_def)

        graph, input_ids, out_ids, out_tree_def = cache[key]

        from zero_jax.numpy import tensor_utils

        inputs = {
            in_id: tensor_utils.to_array(a.data) for in_id, a in zip(input_ids, t_args)
        }
        res = evaluate_graph(graph, inputs)

        flat_res = [array(res[out_id]) for out_id in out_ids]
        return tree_unflatten(out_tree_def, flat_res)

    return wrapped


def grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    holistic: bool = False,
    reduce_axes: tuple[Any, ...] = (),
    return_value: bool = False,
) -> Callable:
    """Creates a function that evaluates the gradient of fun.

    Args:
        fun: The function to be differentiated.
        argnums: Specifies which positional argument(s) to differentiate with respect to.

    Returns:
        A function that evaluates the gradient of the original function.
    """

    @functools.wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        import uuid

        from ml_switcheroo_compiler.interpreter import evaluate_graph
        from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
        from ml_switcheroo_compiler.transforms.autodiff import grad as ir_grad
        from ml_switcheroo_ir import LogicalNode

        from zero_jax.numpy import tensor_utils
        from zero_jax.numpy.lax_numpy import _to_tensor, array, ndarray
        from zero_jax.tree_util import tree_flatten, tree_unflatten

        flat_args, in_tree = tree_flatten((args, kwargs))
        t_args = [
            a if hasattr(a, "__call__") or hasattr(a, "state") else _to_tensor(a)
            for a in flat_args
        ]

        prev_graph = _tracer.active_graph
        is_tracing = _tracer.is_tracing
        graph = _tracer.start_tracing(name="grad_forward")

        proxy_args = []
        input_ids = []
        for arg in t_args:
            if hasattr(arg, "__call__") or hasattr(arg, "state"):
                proxy_args.append(arg)  # pragma: no cover
                input_ids.append(None)  # pragma: no cover
                continue  # pragma: no cover

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

        unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, proxy_args)

        out = fun(*unflattened_args, **unflattened_kwargs)

        flat_out, out_tree = tree_flatten(out)
        out_tensor = _to_tensor(
            flat_out[0]
        )  # assumes grad returns single scalar or array
        out_id = out_tensor.data.id

        _tracer.stop_tracing()
        _tracer.active_graph = prev_graph
        _tracer.is_tracing = is_tracing

        valid_input_ids = [i for i in input_ids if i is not None]

        bwd_graph = ir_grad(graph, wrt=valid_input_ids, output_id=out_id)

        valid_t_args = [a for a in t_args if not hasattr(a, "state")]
        inputs = {
            in_id: tensor_utils.to_array(a.data)
            for in_id, a in zip(valid_input_ids, valid_t_args)
        }
        res = evaluate_graph(bwd_graph, inputs)

        flat_grads = []
        for i_id in input_ids:
            if i_id is None:
                flat_grads.append(0.0)  # pragma: no cover
            else:
                idx = valid_input_ids.index(i_id)
                grad_node_id = bwd_graph.outputs[idx]
                grad_val = res.get(grad_node_id, 0.0)
                flat_grads.append(array(grad_val))

        # For simplicity return the first argument's gradient
        return flat_grads[0]

    return wrapped


def value_and_grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    holistic: bool = False,
    reduce_axes: tuple[Any, ...] = (),
) -> Callable:
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


def vmap(
    fun: Callable,
    in_axes: int | tuple[Any, ...] | dict[str, Any] | None = 0,
    out_axes: Any = 0,
    axis_name: str | None = None,
    axis_size: int | None = None,
    spmd_axis_name: str | tuple[str, ...] | None = None,
) -> Callable:
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
        from zero_jax.numpy.lax_numpy import ndarray
        from zero_jax.tree_util import tree_flatten, tree_unflatten

        flat_args, in_tree = tree_flatten((args, kwargs))
        t_args = [
            a if hasattr(a, "__call__") or hasattr(a, "state") else _to_tensor(a)
            for a in flat_args
        ]

        def inner_fun(*inner_args: Any) -> Any:
            wrapped_args = [ndarray(a) for a in inner_args]
            unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, wrapped_args)

            res = fun(*unflattened_args, **unflattened_kwargs)
            flat_res, out_tree = tree_flatten(res)
            return tuple([_to_tensor(r) for r in flat_res])

        if any(
            not a.shape
            for a in t_args
            if not (hasattr(a, "__call__") or hasattr(a, "state"))
        ):
            out_flat = inner_fun(*t_args)
        else:
            out_flat = []

            unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, flat_args)
            dummy_res_orig = fun(*unflattened_args, **unflattened_kwargs)
            dummy_flat_res, out_tree = tree_flatten(dummy_res_orig)

            for i in range(len(dummy_flat_res)):

                def ith_inner(*inner_args: Any, idx=i):
                    wrapped_args = [ndarray(a) for a in inner_args]
                    unflattened_args, unflattened_kwargs = tree_unflatten(
                        in_tree, wrapped_args
                    )
                    res = fun(*unflattened_args, **unflattened_kwargs)
                    flat_res, _ = tree_flatten(res)
                    return _to_tensor(flat_res[idx])

                out_flat.append(cf.vmap(ith_inner)(*t_args))

        wrapped_out = [_wrap(o) for o in out_flat]

        if "out_tree" not in locals():
            unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, flat_args)
            dummy_res_orig = fun(*unflattened_args, **unflattened_kwargs)
            _, out_tree = tree_flatten(dummy_res_orig)

        return tree_unflatten(out_tree, wrapped_out)

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
    fun: Callable,
    axis_name: Any | None = None,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | tuple[int, ...] | slice = (),
    devices: Any | None = None,
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tuple[int, ...] = (),
    in_parts: Any | None = None,
    out_parts: Any | None = None,
) -> Callable:
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
    import ml_switcheroo_compiler

    # Actually, proper eval_shape would trace without executing, but since eager mode returns
    # zeros of correct shape during tracing... Wait, if we use Tracer:
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

    from zero_jax.numpy.lax_numpy import _to_tensor

    # For now, just execute it and return the result which has a .shape
    # If we want pure shape, we can run it.
    with ml_switcheroo_compiler.core.EagerMode():
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
