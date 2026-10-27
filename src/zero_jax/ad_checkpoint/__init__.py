"""Activation checkpointing (rematerialization) and residual saving policies for zero_jax."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Sequence

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state
from ml_switcheroo_ir import LogicalNode

from zero_jax._types import ConfigValue, PyTree, R, VariadicCallable
from zero_jax.numpy.lax_numpy import ndarray
from zero_jax.tree_util import tree_map

_SAVED_RESIDUAL_NAMES: list[str] = []
_ACTIVE_POLICY_STACK: list[VariadicCallable[bool]] = []


def everything_saveable(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
    """Policy predicate indicating that all intermediates can be saved as residuals.

    Args:
        *args: Positional arguments describing the primitive or checkpoint name.
        **kwargs: Keyword metadata describing the primitive or checkpoint name.

    Returns:
        bool: Always ``True``.
    """
    _ = (args, kwargs)
    return True


def nothing_saveable(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
    """Policy predicate indicating that no intermediates should be saved (recompute all).

    Args:
        *args: Positional arguments describing the primitive or checkpoint name.
        **kwargs: Keyword metadata describing the primitive or checkpoint name.

    Returns:
        bool: Always ``False``.
    """
    _ = (args, kwargs)
    return False


def checkpoint_dots(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
    """Policy predicate saving matrix multiplication (dot) operations as residuals.

    Args:
        *args: Positional arguments (such as operation or checkpoint name).
        **kwargs: Keyword metadata (such as ``name``).

    Returns:
        bool: ``True`` if the operation or checkpoint name represents a dot product.
    """
    target = str(kwargs.get("name", args[0] if args else "")).lower()
    return "dot" in target or "matmul" in target


dots_saveable = checkpoint_dots


def checkpoint_dots_with_no_batch_dims(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
    """Policy predicate saving only dot operations that have no batch dimensions.

    Args:
        *args: Positional arguments (such as operation or checkpoint name).
        **kwargs: Keyword metadata (such as ``name`` and ``batch_dims``).

    Returns:
        bool: ``True`` if the operation is a dot product with zero batch dimensions.
    """
    batch_dims = kwargs.get("batch_dims", 0)
    has_no_batch = batch_dims in (0, (), None)
    return checkpoint_dots(*args, **kwargs) and has_no_batch


dots_with_no_batch_dims_saveable = checkpoint_dots_with_no_batch_dims


def save_only_these_names(*names_to_save: str) -> VariadicCallable[bool]:
    """Return a policy predicate that saves only intermediates tagged with ``names_to_save``.

    Args:
        *names_to_save: Checkpoint names permitted to be saved as residuals.

    Returns:
        VariadicCallable[bool]: Policy predicate function.
    """
    allowed = frozenset(names_to_save)

    def _policy(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
        """Check whether the checkpoint name is in ``allowed``.

        Args:
            *args: Positional arguments (first argument checked as name).
            **kwargs: Keyword arguments (``name`` checked if present).

        Returns:
            bool: ``True`` if the target name is in ``allowed``.
        """
        candidate = kwargs.get("name", args[0] if args else None)
        return isinstance(candidate, str) and candidate in allowed

    return _policy


def save_any_names_but_these(*names_not_to_save: str) -> VariadicCallable[bool]:
    """Return a policy predicate that saves any intermediate except ``names_not_to_save``.

    Args:
        *names_not_to_save: Checkpoint names that must be recomputed rather than saved.

    Returns:
        VariadicCallable[bool]: Policy predicate function.
    """
    excluded = frozenset(names_not_to_save)

    def _policy(*args: ConfigValue, **kwargs: ConfigValue) -> bool:
        """Check whether the checkpoint name is not in ``excluded``.

        Args:
            *args: Positional arguments (first argument checked as name).
            **kwargs: Keyword arguments (``name`` checked if present).

        Returns:
            bool: ``True`` if the target name is not in ``excluded``.
        """
        candidate = kwargs.get("name", args[0] if args else None)
        return not (isinstance(candidate, str) and candidate in excluded)

    return _policy


class _CheckpointPoliciesNamespace:
    """Namespace exposing standard activation checkpointing policies."""

    everything_saveable = staticmethod(everything_saveable)
    nothing_saveable = staticmethod(nothing_saveable)
    checkpoint_dots = staticmethod(checkpoint_dots)
    dots_saveable = staticmethod(dots_saveable)
    checkpoint_dots_with_no_batch_dims = staticmethod(checkpoint_dots_with_no_batch_dims)
    dots_with_no_batch_dims_saveable = staticmethod(dots_with_no_batch_dims_saveable)
    save_only_these_names = staticmethod(save_only_these_names)
    save_any_names_but_these = staticmethod(save_any_names_but_these)


checkpoint_policies = _CheckpointPoliciesNamespace()

_STRING_POLICIES: dict[str, VariadicCallable[bool]] = {
    "everything_saveable": everything_saveable,
    "nothing_saveable": nothing_saveable,
    "checkpoint_dots": checkpoint_dots,
    "dots_saveable": dots_saveable,
    "checkpoint_dots_with_no_batch_dims": checkpoint_dots_with_no_batch_dims,
    "dots_with_no_batch_dims_saveable": dots_with_no_batch_dims_saveable,
}


def _resolve_policy(policy: str | VariadicCallable[bool] | None) -> VariadicCallable[bool]:
    """Resolve an optional policy specification into a callable predicate.

    Args:
        policy: Policy callable, policy name string, or ``None``.

    Returns:
        VariadicCallable[bool]: Resolved policy predicate callable.
    """
    if callable(policy):
        return policy
    if isinstance(policy, str):
        return _STRING_POLICIES.get(policy, everything_saveable)
    return everything_saveable


def checkpoint_name(x: PyTree, name: str) -> PyTree:
    """Tag an intermediate PyTree ``x`` with ``name`` for activation checkpointing policies.

    When tracing is active in ``ml_switcheroo_compiler``, inserts a ``CheckpointName``
    node into the trace graph so compiler passes and residual policies can filter by ``name``.

    Args:
        x: Intermediate array or PyTree of arrays.
        name: Semantic checkpoint identifier string.

    Returns:
        PyTree: The tagged PyTree ``x``.
    """
    active_policy = _ACTIVE_POLICY_STACK[-1] if _ACTIVE_POLICY_STACK else everything_saveable
    saveable = active_policy(name, name=name)
    if saveable:
        _SAVED_RESIDUAL_NAMES.append(name)

    if global_tracing_state.is_tracing:

        def _tag_leaf(leaf: PyTree) -> PyTree:
            """Attach a ``CheckpointName`` trace node to an ``ndarray`` leaf.

            Args:
                leaf: Leaf value from ``x``.

            Returns:
                PyTree: Tagged ``ndarray`` or unchanged non-array leaf.
            """
            if isinstance(leaf, ndarray):
                t = leaf._tensor
                in_id = t.data.id if isinstance(t.data, ProxyTensor) else str(uuid.uuid4())
                out_id = str(uuid.uuid4())
                node = LogicalNode(
                    id=out_id,
                    op_type="CheckpointName",
                    inputs=[in_id],
                    attributes={"name": name, "saveable": saveable},
                    shape_metadata=t.shape,
                )
                global_tracing_state.add_node(node)
                proxy = ProxyTensor(id=out_id, shape=t.shape, dtype=t.dtype.value)
                return ndarray(Tensor(data=proxy, config=TensorConfig(shape=t.shape, dtype=t.dtype, device=t.device)))
            return leaf

        return tree_map(_tag_leaf, x)
    return x


def get_saved_checkpoint_names() -> list[str]:
    """Return the list of checkpoint names saved under the active policy and clear the buffer.

    Returns:
        list[str]: Saved checkpoint names since the previous query.
    """
    saved = list(_SAVED_RESIDUAL_NAMES)
    _SAVED_RESIDUAL_NAMES.clear()
    return saved


def checkpoint(
    fun: VariadicCallable[R] | None = None,
    *,
    prevent_cse: bool = True,
    policy: str | VariadicCallable[bool] | None = None,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Sequence[str] = (),
) -> VariadicCallable[R]:
    """Configure ``fun`` to recompute intermediates during reverse-mode autodiff.

    Supports both direct decoration (``@checkpoint``) and keyword-parameterized
    decoration (``@checkpoint(prevent_cse=True, policy=..., static_argnums=...)``).

    Args:
        fun: Function to rematerialize, or ``None`` when parameterized with keyword arguments.
        prevent_cse: Whether to prevent common subexpression elimination across the boundary.
        policy: Optional rematerialization policy predicate or policy name.
        static_argnums: Optional positional argument index or sequence of indices treated as static.
        static_argnames: Optional keyword argument name or sequence of names treated as static.

    Returns:
        VariadicCallable[R]: Rematerialized callable or decorator.
    """
    _ = (prevent_cse, static_argnums, static_argnames)
    resolved_policy = _resolve_policy(policy)

    def _wrap_callable(target_fun: VariadicCallable[R]) -> VariadicCallable[R]:
        """Wrap ``target_fun`` in a policy-aware activation checkpoint scope.

        Args:
            target_fun: Callable to wrap.

        Returns:
            VariadicCallable[R]: Wrapped callable.
        """

        @functools.wraps(target_fun)
        def _checkpointed(*args: PyTree, **kwargs: PyTree) -> R:
            """Execute ``target_fun`` under the configured checkpointing policy.

            Args:
                *args: Positional arguments forwarded to ``target_fun``.
                **kwargs: Keyword arguments forwarded to ``target_fun``.

            Returns:
                R: Output of ``target_fun(*args, **kwargs)``.
            """
            _ACTIVE_POLICY_STACK.append(resolved_policy)
            try:
                return target_fun(*args, **kwargs)
            finally:
                _ACTIVE_POLICY_STACK.pop()

        return _checkpointed

    if fun is None:
        return _wrap_callable
    return _wrap_callable(fun)


remat = checkpoint

__all__ = [
    "_CheckpointPoliciesNamespace",
    "checkpoint",
    "checkpoint_dots",
    "checkpoint_dots_with_no_batch_dims",
    "checkpoint_name",
    "checkpoint_policies",
    "dots_saveable",
    "dots_with_no_batch_dims_saveable",
    "everything_saveable",
    "get_saved_checkpoint_names",
    "nothing_saveable",
    "remat",
    "save_any_names_but_these",
    "save_only_these_names",
]
