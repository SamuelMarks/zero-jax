"""Transformations for zero_jax."""

from __future__ import annotations

import contextlib
import functools
import uuid
from collections.abc import Callable, Iterator, Sequence

import ml_switcheroo_compiler
import ml_switcheroo_compiler.ops.control_flow as cf
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff import grad as ir_grad
from ml_switcheroo_ir import DType, LogicalGraph, LogicalNode

import zero_jax._compiler_proxy_ops as ops
from zero_jax._config_internal import config as _internal_config
from zero_jax._types import ArrayProtocol, AxisSpec, PyTree, ShapeTuple, StatefulProtocol, VariadicCallable
from zero_jax.lax.control_flow import get_stop_gradient_sources
from zero_jax.numpy import moveaxis, ones_like, tensor_utils, zeros
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, array, ndarray
from zero_jax.sharding import Sharding
from zero_jax.stages import Lowered, Traced
from zero_jax.tree_util import tree_flatten, tree_map, tree_unflatten
from zero_jax.tree_util.pytree import PyTreeDef


class ShapedArray:
    """Wrapper containing abstract shape and dtype information.

    Attributes:
        shape: The shape tuple of the array.
        dtype: The dtype of the array.
    """

    def __init__(self, shape: ShapeTuple, dtype: DType | str) -> None:
        """Initialize ``ShapedArray``.

        Args:
            shape: The shape tuple of the array.
            dtype: The dtype of the array.
        """
        self.shape = shape
        self.dtype = dtype


class _JitWrapped:
    """Callable wrapper produced by :func:`jit` supporting ``.trace()`` and ``.lower()``."""

    def __init__(
        self,
        fun: VariadicCallable[PyTree],
        exec_fn: VariadicCallable[PyTree],
    ) -> None:
        """Initialize ``_JitWrapped``.

        Args:
            fun: Original Python callable.
            exec_fn: Compiled execution callable.
        """
        self._fun = fun
        self._exec_fn = exec_fn
        functools.update_wrapper(self, fun)

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> PyTree:
        """Execute the JIT-compiled function.

        Args:
            *args: Positional PyTree arguments.
            **kwargs: Keyword PyTree arguments.

        Returns:
            PyTree: Evaluated output PyTree.
        """
        if bool(_internal_config.read("jax_disable_jit")):
            return self._fun(*args, **kwargs)
        return self._exec_fn(*args, **kwargs)

    def trace(self, *args: PyTree, **kwargs: PyTree) -> Traced:
        """Return a :class:`zero_jax.stages.Traced` stage for sample arguments.

        Args:
            *args: Sample positional arguments.
            **kwargs: Sample keyword arguments.

        Returns:
            Traced: ``stages.Traced`` object.
        """
        _ = (args, kwargs)
        return Traced(fun=self._exec_fn)

    def lower(self, *args: PyTree, **kwargs: PyTree) -> Lowered:
        """Return a :class:`zero_jax.stages.Lowered` stage for sample arguments.

        Args:
            *args: Sample positional arguments.
            **kwargs: Sample keyword arguments.

        Returns:
            Lowered: ``stages.Lowered`` object.
        """
        _ = (args, kwargs)
        return Lowered(fun=self._exec_fn)


def jit(
    fun: VariadicCallable[PyTree],
    in_shardings: Sharding | Sequence[Sharding | None] | None = None,
    out_shardings: Sharding | Sequence[Sharding | None] | None = None,
    static_argnums: int | Sequence[int] | None = None,
    compiler_options: dict[str, str | int | float | bool] | None = None,
    donate_argnames: str | Sequence[str] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    device: str | None = None,
    inline: bool = False,
    keep_unused: bool = False,
    donate_argnums: int | Sequence[int] | None = None,
    backend: str | None = None,
    abstracted_axes: Sequence[str] | dict[int, str] | None = None,
) -> VariadicCallable[PyTree]:
    """Compile a function for trace-based graph execution.

    Args:
        fun: The function to be JIT-compiled.
        in_shardings: Optional input sharding specifications.
        out_shardings: Optional output sharding specifications.
        static_argnums: Optional positional argument indices treated as static.
        compiler_options: Optional dictionary of compiler options.
        donate_argnames: Optional keyword argument names whose buffers may be donated.
        static_argnames: Optional keyword argument names treated as static.
        device: Optional target device for execution.
        inline: Whether to inline the compiled function into callers.
        keep_unused: Whether to retain unused arguments in the compiled signature.
        donate_argnums: Optional positional argument indices whose buffers may be donated.
        backend: Optional backend identifier string.
        abstracted_axes: Optional specification of abstracted dynamic axes.

    Returns:
        VariadicCallable[PyTree]: A JIT-compiled wrapper around ``fun``.
    """
    _ = (in_shardings, out_shardings, compiler_options, donate_argnames, device, inline, keep_unused, donate_argnums, backend, abstracted_axes)

    static_nums = {static_argnums} if isinstance(static_argnums, int) else set(static_argnums or ())
    static_names = {static_argnames} if isinstance(static_argnames, str) else set(static_argnames or ())
    cache: dict[tuple[tuple[str, ...], tuple[tuple[ShapeTuple, str], ...]], tuple[LogicalGraph, list[str], list[str], PyTreeDef]] = {}

    @functools.wraps(fun)
    def wrapped(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Execute the JIT-compiled function on the provided arguments.

        Args:
            *args: Positional PyTree arguments.
            **kwargs: Keyword PyTree arguments.

        Returns:
            PyTree: The evaluated result PyTree.
        """
        static_pos = {i: args[i] for i in static_nums if i < len(args)}
        dyn_pos = [args[i] for i in range(len(args)) if i not in static_nums]
        static_kw = {k: kwargs[k] for k in static_names if k in kwargs}
        dyn_kw = {k: v for k, v in kwargs.items() if k not in static_names}

        flat_args, tree_def = tree_flatten((dyn_pos, dyn_kw))
        if not flat_args or any(callable(a) or isinstance(a, (str, StatefulProtocol)) for a in flat_args):
            return fun(*args, **kwargs)

        static_key = tuple(f"p{i}={v!r}" for i, v in sorted(static_pos.items())) + tuple(f"k{k}={v!r}" for k, v in sorted(static_kw.items()))
        t_args = [_to_tensor(a) for a in flat_args]
        key = (static_key, tuple((t.shape, t.dtype.value) for t in t_args))

        if key not in cache:
            prev_graph = _tracer.active_graph
            is_tracing = _tracer.is_tracing
            graph = _tracer.start_tracing(name="jit_fun")

            proxy_args = []
            input_ids = []
            for arg in t_args:
                in_id = str(uuid.uuid4())
                input_ids.append(in_id)
                node = LogicalNode(
                    id=in_id,
                    op_type="Input",
                    inputs=[],
                    shape_metadata=arg.shape,
                )
                graph.nodes[in_id] = node
                proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
                proxy_tensor = Tensor(
                    data=proxy,
                    config=TensorConfig(
                        shape=arg.shape,
                        dtype=arg.dtype,
                        device=arg.device,
                    ),
                )
                proxy_args.append(ndarray(proxy_tensor))

            unflattened_dyn_pos, unflattened_dyn_kw = tree_unflatten(tree_def, proxy_args)
            dyn_iter = iter(unflattened_dyn_pos)
            full_args = [static_pos[i] if i in static_pos else next(dyn_iter) for i in range(len(args))]
            full_kwargs = {**unflattened_dyn_kw, **static_kw}
            out = fun(*full_args, **full_kwargs)

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
        inputs = {in_id: tensor_utils.to_array(a.data) for in_id, a in zip(input_ids, t_args)}
        res = evaluate_graph(graph, inputs)
        flat_res = [array(res[out_id]) for out_id in out_ids]
        return tree_unflatten(out_tree_def, flat_res)

    return _JitWrapped(fun, wrapped)


def grad(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[str] = (),
) -> VariadicCallable[PyTree]:
    """Create a function that evaluates the gradient of ``fun``.

    Args:
        fun: The function to be differentiated.
        argnums: Positional argument index or indices to differentiate with respect to.
        has_aux: Whether ``fun`` returns a ``(output, aux)`` pair.
        holomorphic: Whether to differentiate with respect to complex holomorphic inputs.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        reduce_axes: Axis names to reduce over during differentiation.

    Returns:
        VariadicCallable[PyTree]: A function that computes the gradient of ``fun``.
    """
    _ = (holomorphic, allow_int, reduce_axes)

    @functools.wraps(fun)
    def wrapped(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Evaluate the gradient of ``fun`` at the given arguments.

        Args:
            *args: Positional PyTree arguments.
            **kwargs: Keyword PyTree arguments.

        Returns:
            PyTree: The computed gradient (or ``(grad, aux)`` if ``has_aux=True``).
        """
        from zero_jax.api.ad import CustomJVPWrapper, CustomVJPWrapper

        if isinstance(fun, CustomVJPWrapper) and fun.fwd is not None and fun.bwd is not None:
            primal_out, residuals = fun.fwd(*args)
            cot = ones_like(primal_out)
            nondiff = [args[i] for i in fun.nondiff_argnums if i < len(args)]
            bwd_grads = fun.bwd(*nondiff, residuals, cot) if nondiff else fun.bwd(residuals, cot)
            g_tuple = bwd_grads if isinstance(bwd_grads, tuple) else (bwd_grads,)
            if isinstance(argnums, int):
                return g_tuple[argnums] if argnums < len(g_tuple) else g_tuple[0]
            return tuple(g_tuple[i] for i in argnums)

        if isinstance(fun, CustomJVPWrapper) and fun.jvp_fun is not None:
            nondiff = [args[i] for i in fun.nondiff_argnums if i < len(args)]
            diff_args = tuple(a for i, a in enumerate(args) if i not in fun.nondiff_argnums)
            tangents = tuple(ones_like(a) for a in diff_args)
            _, tangent_out = fun.jvp_fun(*nondiff, diff_args, tangents) if nondiff else fun.jvp_fun(diff_args, tangents)
            return tangent_out

        flat_args, in_tree = tree_flatten((args, kwargs))
        t_args = [a if callable(a) or isinstance(a, StatefulProtocol) else _to_tensor(a) for a in flat_args]

        prev_graph = _tracer.active_graph
        is_tracing = _tracer.is_tracing
        graph = _tracer.start_tracing(name="grad_forward")

        proxy_args = []
        input_ids: list[str | None] = []
        for arg in t_args:
            if callable(arg) or isinstance(arg, StatefulProtocol):
                proxy_args.append(arg)
                input_ids.append(None)
                continue

            in_id = str(uuid.uuid4())
            input_ids.append(in_id)
            node = LogicalNode(
                id=in_id,
                op_type="Input",
                inputs=[],
                shape_metadata=arg.shape,
            )
            graph.nodes[in_id] = node
            proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
            proxy_tensor = Tensor(
                data=proxy,
                config=TensorConfig(
                    shape=arg.shape,
                    dtype=arg.dtype,
                    device=arg.device,
                ),
            )
            proxy_args.append(ndarray(proxy_tensor))

        unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, proxy_args)
        out = fun(*unflattened_args, **unflattened_kwargs)
        if has_aux and isinstance(out, tuple) and len(out) == 2:
            primary_out, _ = out
        else:
            primary_out = out

        flat_out, _ = tree_flatten(primary_out)
        if len(flat_out) > 1:
            acc_t = ops.sum(_to_tensor(flat_out[0]))
            for extra_leaf in flat_out[1:]:
                acc_t = ops.add(acc_t, ops.sum(_to_tensor(extra_leaf)))
            out_tensor = acc_t
        else:
            out_tensor = _to_tensor(flat_out[0])
        out_id = out_tensor.data.id

        _tracer.stop_tracing()
        _tracer.active_graph = prev_graph
        _tracer.is_tracing = is_tracing

        valid_input_ids = [i for i in input_ids if i is not None]
        bwd_graph = ir_grad(graph, wrt=valid_input_ids, output_id=out_id)
        valid_t_args = [a for a in t_args if not (callable(a) or isinstance(a, StatefulProtocol))]
        inputs = {in_id: tensor_utils.to_array(a.data) for in_id, a in zip(valid_input_ids, valid_t_args)}
        for sg_id, src_id in get_stop_gradient_sources(id(graph)).items():
            inputs[sg_id] = inputs[src_id]
        res = evaluate_graph(bwd_graph, inputs)

        flat_grads: list[PyTree] = []
        for i_id in input_ids:
            if i_id is None:
                flat_grads.append(0.0)
            else:
                idx = valid_input_ids.index(i_id)
                grad_node_id = bwd_graph.outputs[idx]
                grad_val = res.get(grad_node_id, 0.0)
                flat_grads.append(array(grad_val))

        grad_pos_args, _ = tree_unflatten(in_tree, flat_grads)
        selected_grad = grad_pos_args[argnums] if isinstance(argnums, int) else tuple(grad_pos_args[i] for i in argnums)

        if has_aux:
            eager_out = fun(*args, **kwargs)
            aux_val = eager_out[1] if isinstance(eager_out, tuple) and len(eager_out) == 2 else None
            return selected_grad, aux_val
        return selected_grad

    return wrapped


def value_and_grad(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[str] = (),
) -> VariadicCallable[tuple[PyTree, PyTree]]:
    """Create a function that evaluates both the value and gradient of ``fun``.

    Args:
        fun: The function to be differentiated.
        argnums: Positional argument index or indices to differentiate with respect to.
        has_aux: Whether ``fun`` returns auxiliary data.
        holomorphic: Whether to compute holomorphic gradients.
        allow_int: Whether to allow integer differentiation.
        reduce_axes: Axis names to reduce over.

    Returns:
        VariadicCallable[tuple[PyTree, PyTree]]: A function returning ``(value, grad)``.
    """

    @functools.wraps(fun)
    def wrapped(*args: PyTree, **kwargs: PyTree) -> tuple[PyTree, PyTree]:
        """Execute ``fun`` and compute its value and gradient.

        Args:
            *args: Positional PyTree arguments.
            **kwargs: Keyword PyTree arguments.

        Returns:
            tuple[PyTree, PyTree]: Tuple of ``(value, gradient)``.
        """
        val = fun(*args, **kwargs)
        g = grad(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int)(*args, **kwargs)
        if has_aux:
            return val, g[0]
        return val, g

    return wrapped


def vmap(
    fun: VariadicCallable[PyTree],
    in_axes: int | Sequence[AxisSpec] | dict[str, AxisSpec] | None = 0,
    axis_name: str | None = None,
    axis_size: int | None = None,
    out_axes: AxisSpec = 0,
    spmd_axis_name: str | tuple[str, ...] | None = None,
    sum_match: bool = False,
) -> VariadicCallable[PyTree]:
    """Create a function which maps ``fun`` over argument axes.

    Args:
        fun: The function to be mapped over argument axes.
        in_axes: Specification of input axes to map over.
        axis_name: Optional name for the mapped axis.
        axis_size: Optional explicit size of the mapped axis.
        out_axes: Specification of where the mapped axis appears in the output.
        spmd_axis_name: Optional SPMD axis name or tuple of names.
        sum_match: Whether to match summed axes during reverse-mode AD.

    Returns:
        VariadicCallable[PyTree]: A vectorized version of ``fun``.
    """
    _ = (axis_name, axis_size, spmd_axis_name, sum_match)

    @functools.wraps(fun)
    def wrapped(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Execute the vectorized function over the provided batched arguments.

        Args:
            *args: Positional PyTree arguments.
            **kwargs: Keyword PyTree arguments.

        Returns:
            PyTree: The vectorized output PyTree.
        """
        norm_args = list(args)
        if isinstance(in_axes, int) and in_axes != 0:
            norm_args = [tree_map(lambda x: moveaxis(x, in_axes, 0) if isinstance(x, ArrayProtocol) and x.ndim > 0 else x, a) for a in norm_args]
        elif isinstance(in_axes, (list, tuple)):
            for idx, ax in enumerate(in_axes):
                if idx < len(norm_args) and isinstance(ax, int) and ax != 0:
                    norm_args[idx] = tree_map(lambda x, _a=ax: moveaxis(x, _a, 0) if isinstance(x, ArrayProtocol) and x.ndim > 0 else x, norm_args[idx])

        flat_args, in_tree = tree_flatten((tuple(norm_args), kwargs))
        t_args = [a if callable(a) or isinstance(a, StatefulProtocol) else _to_tensor(a) for a in flat_args]

        unflattened_args_orig, unflattened_kwargs_orig = tree_unflatten(in_tree, flat_args)
        dummy_res_orig = fun(*unflattened_args_orig, **unflattened_kwargs_orig)
        dummy_flat_res, out_tree = tree_flatten(dummy_res_orig)

        def inner_fun(*inner_args: Tensor) -> tuple[Tensor, ...]:
            """Evaluate ``fun`` on a single unbatched slice.

            Args:
                *inner_args: Unbatched tensor arguments.

            Returns:
                tuple[Tensor, ...]: Flattened output tensors.
            """
            wrapped_args = [ndarray(a) for a in inner_args]
            unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, wrapped_args)
            res = fun(*unflattened_args, **unflattened_kwargs)
            flat_res, _ = tree_flatten(res)
            return tuple(_to_tensor(r) for r in flat_res)

        if any(not a.shape for a in t_args if not (callable(a) or isinstance(a, StatefulProtocol))):
            out_flat = list(inner_fun(*t_args))
        else:
            out_flat = []
            for i in range(len(dummy_flat_res)):

                def ith_inner(*inner_args: Tensor, idx: int = i) -> Tensor:
                    """Evaluate the ``idx``-th output leaf of ``fun``.

                    Args:
                        *inner_args: Unbatched tensor arguments.
                        idx: Output leaf index.

                    Returns:
                        Tensor: The ``idx``-th output tensor.
                    """
                    wrapped_args = [ndarray(a) for a in inner_args]
                    unflattened_args, unflattened_kwargs = tree_unflatten(in_tree, wrapped_args)
                    res = fun(*unflattened_args, **unflattened_kwargs)
                    flat_res, _ = tree_flatten(res)
                    return _to_tensor(flat_res[idx])

                out_flat.append(cf.vmap(ith_inner)(*t_args))

        wrapped_out = [moveaxis(_wrap(o), 0, out_axes) if isinstance(out_axes, int) and out_axes != 0 else _wrap(o) for o in out_flat]
        return tree_unflatten(out_tree, wrapped_out)

    return wrapped


@contextlib.contextmanager
def disable_jit(disable: bool = True) -> Iterator[None]:
    """Context manager to temporarily disable JIT compilation.

    Args:
        disable: Whether to disable JIT compilation inside the context.

    Yields:
        None: Context where JIT compilation is disabled.
    """
    prev = bool(_internal_config.read("jax_disable_jit"))
    _internal_config.update("jax_disable_jit", disable)
    try:
        yield
    finally:
        _internal_config.update("jax_disable_jit", prev)


def pmap(
    fun: VariadicCallable[PyTree],
    axis_name: str | None = None,
    in_axes: int | Sequence[AxisSpec] | None = 0,
    out_axes: AxisSpec = 0,
    static_broadcasted_argnums: int | Sequence[int] = (),
    devices: Sequence[str] | None = None,
    axis_size: int | None = None,
    backend: str | None = None,
    donate_argnums: int | Sequence[int] = (),
) -> VariadicCallable[PyTree]:
    """Create a parallel mapped function across devices.

    Args:
        fun: The function to be mapped in parallel.
        axis_name: Optional name of the mapped axis.
        in_axes: Specification of input axes to map over.
        out_axes: Specification of output axes.
        static_broadcasted_argnums: Argument indices to broadcast statically.
        devices: Optional sequence of target devices.
        axis_size: Optional size of the mapped axis.
        backend: Optional backend identifier string.
        donate_argnums: Argument indices whose buffers can be donated.

    Returns:
        VariadicCallable[PyTree]: A parallelized version of ``fun``.
    """
    _ = (static_broadcasted_argnums, devices, backend, donate_argnums)
    return vmap(fun, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name, axis_size=axis_size)


def eval_shape(fun: VariadicCallable[PyTree], *args: PyTree, **kwargs: PyTree) -> PyTree:
    """Evaluate the output shape and dtype PyTree of ``fun`` without materializing final buffers.

    Args:
        fun: The function whose output shape is evaluated.
        *args: Positional arguments (including ``ShapeDtypeStruct``) passed to ``fun``.
        **kwargs: Keyword arguments passed to ``fun``.

    Returns:
        PyTree: PyTree of :class:`ShapedArray` objects containing ``shape`` and ``dtype``.
    """

    def _materialize_abstract(leaf: PyTree) -> PyTree:
        """Convert abstract ``ShapeDtypeStruct`` or ``ShapedArray`` into a zero tensor for tracing.

        Args:
            leaf: Input PyTree leaf.

        Returns:
            PyTree: Concrete array or scalar leaf.
        """
        if isinstance(leaf, ArrayProtocol):
            return zeros(leaf.shape, dtype=leaf.dtype)
        return leaf

    concrete_args = tree_map(_materialize_abstract, args)
    concrete_kwargs = tree_map(_materialize_abstract, kwargs)

    with ml_switcheroo_compiler.core.EagerMode():
        res = fun(*concrete_args, **concrete_kwargs)

    def _to_shaped_array(leaf: PyTree) -> ShapedArray:
        """Convert concrete output leaf to ``ShapedArray``.

        Args:
            leaf: Output leaf.

        Returns:
            ShapedArray: Abstract shape and dtype wrapper.
        """
        t = _to_tensor(leaf)
        return ShapedArray(t.shape, t.dtype)

    return tree_map(_to_shaped_array, res)


__all__ = [
    "ShapedArray",
    "_JitWrapped",
    "disable_jit",
    "eval_shape",
    "grad",
    "jit",
    "pmap",
    "value_and_grad",
    "vmap",
]
