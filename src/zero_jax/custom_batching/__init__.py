"""Frontend API routing for jax.custom_batching."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable


class custom_vmap:
    """Decorator class for defining custom batching (``vmap``) rules."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``custom_vmap`` wrapper.

        Args:
            fun: Function to wrap with a custom ``vmap`` rule.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.fun = fun if callable(fun) else None
        self.vmap_rule: VariadicCallable[ArrayLikeSpec] | None = None
        self.id = id
        self.name = name
        self.value = value

    def def_vmap(self, vmap_rule: VariadicCallable[ArrayLikeSpec]) -> VariadicCallable[ArrayLikeSpec]:
        """Register a custom batching rule for the wrapped function.

        Args:
            vmap_rule: Custom batching rule callable.

        Returns:
            VariadicCallable[ArrayLikeSpec]: The registered ``vmap_rule``.
        """
        self.vmap_rule = vmap_rule
        return vmap_rule

    def __call__(self, *call_args: ArrayLikeSpec, **call_kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the wrapped function.

        Args:
            *call_args: Positional arguments.
            **call_kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Result of ``self.fun(*call_args, **call_kwargs)``.
        """
        if self.fun is not None:
            return self.fun(*call_args, **call_kwargs)
        return None


def sequential_vmap(fun: VariadicCallable[ArrayLikeSpec]) -> VariadicCallable[ArrayLikeSpec]:
    """Vectorize ``fun`` sequentially using :func:`zero_jax.vmap`.

    Args:
        fun: Callable to vectorize.

    Returns:
        VariadicCallable[ArrayLikeSpec]: Vectorized callable.
    """
    from zero_jax.api.transformations import vmap

    return vmap(fun)


__all__ = ["custom_vmap", "sequential_vmap"]
