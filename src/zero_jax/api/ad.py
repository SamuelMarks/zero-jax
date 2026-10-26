"""Automatic Differentiation APIs."""

from __future__ import annotations

import math
from collections.abc import Sequence

from zero_jax._types import PyTree, VariadicCallable
from zero_jax.numpy import reshape, stack, zeros_like
from zero_jax.numpy import sum as jnp_sum
from zero_jax.numpy.lax_numpy import _to_tensor
from zero_jax.tree_util import tree_flatten, tree_unflatten

from .transformations import grad, value_and_grad


class CustomJVPWrapper:
    """Callable wrapper supporting custom JVP rule registration."""

    def __init__(self, fun: VariadicCallable[PyTree], nondiff_argnums: Sequence[int] = ()) -> None:
        """Initialize ``CustomJVPWrapper``.

        Args:
            fun: Wrapped primal function.
            nondiff_argnums: Non-differentiable positional argument indices.
        """
        self.fun = fun
        self.nondiff_argnums = tuple(nondiff_argnums)
        self.jvp_fun: VariadicCallable[PyTree] | None = None

    def defjvp(self, jvp_fun: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
        """Register a custom JVP rule function.

        Args:
            jvp_fun: Custom JVP rule callable ``(primals, tangents) -> (primal_out, tangent_out)``.

        Returns:
            VariadicCallable[PyTree]: The registered ``jvp_fun``.
        """
        self.jvp_fun = jvp_fun
        return jvp_fun

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> PyTree:
        """Invoke the wrapped primal function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Result of calling ``self.fun``.
        """
        return self.fun(*args, **kwargs)


class CustomVJPWrapper:
    """Callable wrapper supporting custom VJP rule registration."""

    def __init__(self, fun: VariadicCallable[PyTree], nondiff_argnums: Sequence[int] = ()) -> None:
        """Initialize ``CustomVJPWrapper``.

        Args:
            fun: Wrapped primal function.
            nondiff_argnums: Non-differentiable positional argument indices.
        """
        self.fun = fun
        self.nondiff_argnums = tuple(nondiff_argnums)
        self.fwd: VariadicCallable[PyTree] | None = None
        self.bwd: VariadicCallable[PyTree] | None = None

    def defvjp(self, fwd: VariadicCallable[PyTree], bwd: VariadicCallable[PyTree]) -> None:
        """Register forward and backward functions for custom VJP.

        Args:
            fwd: Forward pass function returning ``(primal_out, residuals)``.
            bwd: Backward pass function computing cotangents from residuals.
        """
        self.fwd = fwd
        self.bwd = bwd

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> PyTree:
        """Invoke the wrapped primal function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Result of calling ``self.fun``.
        """
        return self.fun(*args, **kwargs)


def custom_jvp(
    fun: VariadicCallable[PyTree],
    nondiff_argnums: Sequence[int] = (),
) -> CustomJVPWrapper:
    """Wrap a function for custom JVP rule definition.

    Args:
        fun: Function to wrap.
        nondiff_argnums: Tuple of non-differentiable argument indices.

    Returns:
        CustomJVPWrapper: Wrapped function exposing ``defjvp``.
    """
    return CustomJVPWrapper(fun, nondiff_argnums=nondiff_argnums)


def custom_vjp(
    fun: VariadicCallable[PyTree],
    nondiff_argnums: Sequence[int] = (),
) -> CustomVJPWrapper:
    """Wrap a function for custom VJP rule definition.

    Args:
        fun: Function to wrap.
        nondiff_argnums: Tuple of non-differentiable argument indices.

    Returns:
        CustomVJPWrapper: Wrapped function exposing ``defvjp``.
    """
    return CustomVJPWrapper(fun, nondiff_argnums=nondiff_argnums)


def custom_gradient(fun: VariadicCallable[PyTree]) -> CustomVJPWrapper:
    """Decorate a function returning ``(primal_out, vjp_fn)`` into a :class:`CustomVJPWrapper`.

    Args:
        fun: Function returning ``(value, grad_fn)`` or plain value.

    Returns:
        CustomVJPWrapper: Wrapped function with custom VJP rule.
    """

    def _primal(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Evaluate primal result from ``fun``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Primal output.
        """
        res = fun(*args, **kwargs)
        if isinstance(res, tuple) and len(res) == 2 and callable(res[1]):
            return res[0]
        return res

    wrapper = CustomVJPWrapper(_primal)

    def _fwd(*args: PyTree) -> tuple[PyTree, PyTree]:
        """Forward rule for ``custom_gradient``.

        Args:
            *args: Primal arguments.

        Returns:
            tuple[PyTree, PyTree]: ``(primal_out, vjp_fn)``.
        """
        res = fun(*args)
        if isinstance(res, tuple) and len(res) == 2 and callable(res[1]):
            return res[0], res[1]
        return res, lambda g: (g,) * len(args)

    def _bwd(vjp_fn: VariadicCallable[PyTree], cotangent: PyTree) -> tuple[PyTree, ...]:
        """Backward rule for ``custom_gradient``.

        Args:
            vjp_fn: Residual VJP callable returned by ``fun``.
            cotangent: Output cotangent.

        Returns:
            tuple[PyTree, ...]: Tuple of input cotangents.
        """
        g_out = vjp_fn(cotangent)
        return g_out if isinstance(g_out, tuple) else (g_out,)

    wrapper.defvjp(_fwd, _bwd)
    return wrapper


def jvp(
    fun: VariadicCallable[PyTree],
    primals: Sequence[PyTree],
    tangents: Sequence[PyTree],
    has_aux: bool = False,
) -> tuple[PyTree, ...]:
    """Compute the forward-mode Jacobian-vector product of ``fun`` at ``primals`` along ``tangents``.

    Args:
        fun: Function to be differentiated.
        primals: Sequence of primal values.
        tangents: Sequence of tangent values matching ``primals``.
        has_aux: Whether ``fun`` returns ``(output, aux)``.

    Returns:
        tuple[PyTree, ...]: ``(primal_out, tangent_out)`` or ``(primal_out, tangent_out, aux)`` when ``has_aux=True``.
    """
    if isinstance(fun, CustomJVPWrapper) and fun.jvp_fun is not None:
        nondiff = [primals[i] for i in fun.nondiff_argnums if i < len(primals)]
        diff_primals = tuple(p for i, p in enumerate(primals) if i not in fun.nondiff_argnums)
        diff_tangents = tuple(t for i, t in enumerate(tangents) if i not in fun.nondiff_argnums)
        return fun.jvp_fun(*nondiff, diff_primals, diff_tangents) if nondiff else fun.jvp_fun(diff_primals, diff_tangents)

    raw_out = fun(*primals)
    if has_aux and isinstance(raw_out, tuple) and len(raw_out) == 2:
        primal_out, aux = raw_out
    else:
        primal_out, aux = raw_out, None

    def _pure_fun(*args: PyTree) -> PyTree:
        """Evaluate primary output of ``fun``.

        Args:
            *args: Primal arguments.

        Returns:
            PyTree: Primary output without auxiliary data.
        """
        r = fun(*args)
        return r[0] if (has_aux and isinstance(r, tuple) and len(r) == 2) else r

    all_argnums = tuple(range(len(primals)))

    def _leaf_jvp(out_leaf: PyTree, leaf_idx: int) -> PyTree:
        """Compute directional derivative for the ``leaf_idx``-th output leaf.

        Args:
            out_leaf: Output leaf value.
            leaf_idx: Flat output leaf index.

        Returns:
            PyTree: Directional derivative matching ``out_leaf``.
        """
        if not primals:
            return out_leaf * 0.0

        def _scalar_comp(*args: PyTree) -> PyTree:
            """Extract and sum the ``leaf_idx``-th leaf of ``_pure_fun(*args)``.

            Args:
                *args: Primal arguments.

            Returns:
                PyTree: Scalar sum of target leaf.
            """
            leaves, _ = tree_flatten(_pure_fun(*args))
            return jnp_sum(leaves[leaf_idx])

        g_vals = grad(_scalar_comp, argnums=all_argnums)(*primals)
        total_tangent: PyTree = None
        for g_val, t_val in zip(g_vals, tangents):
            contrib = jnp_sum(g_val * t_val)
            total_tangent = contrib if total_tangent is None else (total_tangent + contrib)
        return total_tangent

    out_leaves, out_def = tree_flatten(primal_out)
    tangent_leaves = [_leaf_jvp(leaf, idx) for idx, leaf in enumerate(out_leaves)]
    tangent_out = tree_unflatten(out_def, tangent_leaves)
    if has_aux:
        return primal_out, tangent_out, aux
    return primal_out, tangent_out


def vjp(
    fun: VariadicCallable[PyTree],
    *primals: PyTree,
    has_aux: bool = False,
) -> tuple[PyTree, ...]:
    """Compute the reverse-mode vector-Jacobian product of ``fun`` at ``primals``.

    Args:
        fun: Function to be differentiated.
        *primals: Primal argument values.
        has_aux: Whether ``fun`` returns ``(output, aux)``.

    Returns:
        tuple[PyTree, ...]: ``(primal_out, vjp_fun)`` or ``(primal_out, vjp_fun, aux)`` when ``has_aux=True``.
    """
    if isinstance(fun, CustomVJPWrapper) and fun.fwd is not None and fun.bwd is not None:
        fwd_fn = fun.fwd
        bwd_fn = fun.bwd
        primal_out, residuals = fwd_fn(*primals)
        nondiff = [primals[i] for i in fun.nondiff_argnums if i < len(primals)]

        def _custom_vjp_fun(cotangent: PyTree) -> tuple[PyTree, ...]:
            """Evaluate custom VJP backward rule.

            Args:
                cotangent: Output cotangent.

            Returns:
                tuple[PyTree, ...]: Tuple of input cotangents.
            """
            res = bwd_fn(*nondiff, residuals, cotangent) if nondiff else bwd_fn(residuals, cotangent)
            return res if isinstance(res, tuple) else (res,)

        return (primal_out, _custom_vjp_fun, None) if has_aux else (primal_out, _custom_vjp_fun)

    raw_out = fun(*primals)
    if has_aux and isinstance(raw_out, tuple) and len(raw_out) == 2:
        primal_out, aux = raw_out
    else:
        primal_out, aux = raw_out, None

    all_argnums = tuple(range(len(primals)))

    def vjp_fun(cotangent: PyTree) -> tuple[PyTree, ...]:
        """Pull back ``cotangent`` through ``fun`` at ``primals``.

        Args:
            cotangent: Output cotangent matching ``primimal_out``.

        Returns:
            tuple[PyTree, ...]: Tuple of cotangents with respect to ``primals``.
        """
        cot_leaves, _ = tree_flatten(cotangent)

        def _weighted_scalar(*args: PyTree) -> PyTree:
            """Compute inner product of ``fun(*args)`` with ``cotangent``.

            Args:
                *args: Primal arguments.

            Returns:
                PyTree: Scalar inner product.
            """
            r = fun(*args)
            p_out = r[0] if (has_aux and isinstance(r, tuple) and len(r) == 2) else r
            out_leaves, _ = tree_flatten(p_out)
            terms = [jnp_sum(o_l * c_l) for o_l, c_l in zip(out_leaves, cot_leaves)]
            total = terms[0]
            for term in terms[1:]:
                total = total + term
            return total

        return grad(_weighted_scalar, argnums=all_argnums)(*primals)

    if has_aux:
        return primal_out, vjp_fun, aux
    return primal_out, vjp_fun


def linearize(
    fun: VariadicCallable[PyTree],
    *primals: PyTree,
    has_aux: bool = False,
) -> tuple[PyTree, ...]:
    """Produce a linear approximation to ``fun`` at ``primals`` using :func:`jvp`.

    Args:
        fun: Function to be linearized.
        *primals: Primal argument values.
        has_aux: Whether ``fun`` returns ``(output, aux)``.

    Returns:
        tuple[PyTree, ...]: ``(primal_out, jvp_fun)`` or ``(primal_out, jvp_fun, aux)``.
    """
    raw_out = fun(*primals)
    if has_aux and isinstance(raw_out, tuple) and len(raw_out) == 2:
        primal_out, aux = raw_out
    else:
        primal_out, aux = raw_out, None

    def jvp_fun(*tangents: PyTree) -> PyTree:
        """Evaluate the linearized tangent map at ``tangents``.

        Args:
            *tangents: Input tangent vectors matching ``primals``.

        Returns:
            PyTree: Output tangent PyTree.
        """
        res = jvp(fun, primals, tangents, has_aux=has_aux)
        return res[1]

    if has_aux:
        return primal_out, jvp_fun, aux
    return primal_out, jvp_fun


def linear_transpose(
    fun: VariadicCallable[PyTree],
    *primals: PyTree,
) -> VariadicCallable[PyTree]:
    """Transpose a function that is promised to be linear in ``primals``.

    Args:
        fun: Linear callable.
        *primals: Example inputs specifying shape and dtype structure.

    Returns:
        VariadicCallable[PyTree]: Transposed linear function returning a tuple of cotangents.
    """
    if not primals:
        return fun
    _, vjp_fn = vjp(fun, *primals)
    return vjp_fn


def jacfwd(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> VariadicCallable[PyTree]:
    """Compute the Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Argument index or sequence of indices to differentiate with respect to.
        has_aux: Whether ``fun`` returns ``(output, aux)``.
        holomorphic: Whether ``fun`` is holomorphic.

    Returns:
        VariadicCallable[PyTree]: Function that evaluates the Jacobian of ``fun``.
    """
    return jacrev(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic)


def jacrev(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> VariadicCallable[PyTree]:
    """Compute the Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Argument index or sequence of indices to differentiate with respect to.
        has_aux: Whether ``fun`` returns ``(output, aux)``.
        holomorphic: Whether ``fun`` is holomorphic.
        allow_int: Whether integer differentiation is permitted.

    Returns:
        VariadicCallable[PyTree]: Function that evaluates the Jacobian of ``fun``.
    """
    _ = holomorphic

    def wrapper(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Evaluate the Jacobian at ``args`` and ``kwargs``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Evaluated Jacobian tensor or PyTree of Jacobian tensors.
        """
        raw = fun(*args, **kwargs)
        out = raw[0] if (has_aux and isinstance(raw, tuple) and len(raw) == 2) else raw
        t_out = _to_tensor(out)
        if not t_out.shape:
            return grad(fun, argnums=argnums, has_aux=has_aux, allow_int=allow_int)(*args, **kwargs)

        num_elems = int(math.prod(t_out.shape))
        rows = []
        for idx in range(num_elems):

            def _elem_fn(*inner_args: PyTree, _i: int = idx, **inner_kwargs: PyTree) -> PyTree:
                """Select the ``_i``-th flattened element of ``fun`` output.

                Args:
                    *inner_args: Positional arguments.
                    _i: Flat element index.
                    **inner_kwargs: Keyword arguments.

                Returns:
                    PyTree: Scalar element ``_i``.
                """
                r = fun(*inner_args, **inner_kwargs)
                val = r[0] if (has_aux and isinstance(r, tuple) and len(r) == 2) else r
                flat = reshape(val, (num_elems,))
                return flat[_i]

            rows.append(grad(_elem_fn, argnums=argnums, allow_int=allow_int)(*args, **kwargs))

        stacked = stack(rows, axis=0)
        in_idx = argnums if isinstance(argnums, int) else argnums[0]
        in_shape = _to_tensor(args[in_idx]).shape
        jac = reshape(stacked, (*t_out.shape, *in_shape))
        if has_aux and isinstance(raw, tuple) and len(raw) == 2:
            return jac, raw[1]
        return jac

    return wrapper


def jacobian(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> VariadicCallable[PyTree]:
    """Compute the Jacobian of ``fun`` (alias for :func:`jacrev`).

    Args:
        fun: Function to be differentiated.
        argnums: Argument index or indices to differentiate with respect to.
        has_aux: Whether the function returns auxiliary data.
        holomorphic: Whether ``fun`` is holomorphic.

    Returns:
        VariadicCallable[PyTree]: A function that evaluates the Jacobian.
    """
    return jacrev(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic)


def hessian(
    fun: VariadicCallable[PyTree],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> VariadicCallable[PyTree]:
    """Compute the Hessian matrix of ``fun``.

    Args:
        fun: Scalar-valued function to be twice differentiated.
        argnums: Argument index or indices to differentiate with respect to.
        has_aux: Whether the function returns auxiliary data.
        holomorphic: Whether ``fun`` is holomorphic.

    Returns:
        VariadicCallable[PyTree]: A function that evaluates the Hessian.
    """
    grad_fn = grad(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic)

    def wrapper(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Evaluate the Hessian at ``args`` and ``kwargs``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Evaluated Hessian tensor.
        """
        in_idx = argnums if isinstance(argnums, int) else argnums[0]
        x = args[in_idx]
        t_x = _to_tensor(x)
        eps = 1e-4
        if not t_x.shape:
            pos_args_plus = list(args)
            pos_args_minus = list(args)
            pos_args_plus[in_idx] = x + eps
            pos_args_minus[in_idx] = x - eps
            g_plus = grad_fn(*pos_args_plus, **kwargs)
            g_minus = grad_fn(*pos_args_minus, **kwargs)
            if has_aux and isinstance(g_plus, tuple) and len(g_plus) == 2:
                return (g_plus[0] - g_minus[0]) / (2.0 * eps), g_plus[1]
            return (g_plus - g_minus) / (2.0 * eps)

        num_elems = int(math.prod(t_x.shape))
        flat_x = reshape(x, (num_elems,))
        cols = []
        for idx in range(num_elems):
            delta = zeros_like(flat_x).at[idx].set(eps)
            delta_shaped = reshape(delta, t_x.shape)
            pos_args_plus = list(args)
            pos_args_minus = list(args)
            pos_args_plus[in_idx] = x + delta_shaped
            pos_args_minus[in_idx] = x - delta_shaped
            g_plus = grad_fn(*pos_args_plus, **kwargs)
            g_minus = grad_fn(*pos_args_minus, **kwargs)
            gp = g_plus[0] if (has_aux and isinstance(g_plus, tuple) and len(g_plus) == 2) else g_plus
            gm = g_minus[0] if (has_aux and isinstance(g_minus, tuple) and len(g_minus) == 2) else g_minus
            cols.append((gp - gm) / (2.0 * eps))
        stacked = stack(cols, axis=0)
        return reshape(stacked, (*t_x.shape, *t_x.shape))

    return wrapper


__all__ = [
    "CustomJVPWrapper",
    "CustomVJPWrapper",
    "custom_gradient",
    "custom_jvp",
    "custom_vjp",
    "grad",
    "hessian",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jvp",
    "linear_transpose",
    "linearize",
    "value_and_grad",
    "vjp",
]
