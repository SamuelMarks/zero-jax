"""Frontend API routing for jax.custom_transpose."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable


class custom_transpose:
    """Decorator class for defining custom linear transpose rules."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``custom_transpose`` wrapper.

        Args:
            fun: Linear function to wrap.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.fun = fun if callable(fun) else None
        self.transpose_rule: VariadicCallable[ArrayLikeSpec] | None = None
        self.id = id
        self.name = name
        self.value = value

    def def_transpose(self, transpose_rule: VariadicCallable[ArrayLikeSpec]) -> VariadicCallable[ArrayLikeSpec]:
        """Register the custom transpose rule.

        Args:
            transpose_rule: Custom transpose callable.

        Returns:
            VariadicCallable[ArrayLikeSpec]: Registered ``transpose_rule``.
        """
        self.transpose_rule = transpose_rule
        return transpose_rule

    def __call__(self, *call_args: ArrayLikeSpec, **call_kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the wrapped linear function.

        Args:
            *call_args: Positional arguments.
            **call_kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Function output.
        """
        if self.fun is not None:
            return self.fun(*call_args, **call_kwargs)
        return None


__all__ = ["custom_transpose"]
