"""Frontend API routing for jax.stages."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue, DTypeSpec, ShapeTuple, VariadicCallable
from zero_jax.sharding import Sharding


class ArgInfo:
    """Information about a traced or lowered function argument."""

    def __init__(
        self,
        aval: ConfigValue = None,
        donated: bool = False,
        shape: ShapeTuple = (),
        dtype: DTypeSpec = "float32",
    ) -> None:
        """Initialize ``ArgInfo``.

        Args:
            aval: Abstract value descriptor.
            donated: Whether the buffer is donated.
            shape: Shape tuple of the argument.
            dtype: Data type of the argument.
        """
        self._aval = aval
        self.donated = donated
        self.shape = shape
        self.dtype = dtype


class OutInfo:
    """Information about a traced or lowered function output."""

    def __init__(
        self,
        shape: ShapeTuple = (),
        dtype: DTypeSpec = "float32",
        sharding: Sharding | None = None,
    ) -> None:
        """Initialize ``OutInfo``.

        Args:
            shape: Output tensor shape tuple.
            dtype: Output tensor dtype.
            sharding: Optional output sharding specification.
        """
        self.shape = shape
        self.dtype = dtype
        self.sharding = sharding


class CompilerOptions(dict[str, ConfigValue]):
    """Dictionary of compiler options passed during lowering and compilation."""

    def __init__(
        self,
        *args: dict[str, ConfigValue],
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``CompilerOptions``.

        Args:
            *args: Optional initial mapping positional argument.
            **kwargs: Compiler option key-value pairs.
        """
        super().__init__(*args, **kwargs)


class Compiled:
    """Compiled representation of a function specialized to argument types and shapes."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
        args_info: tuple[ArgInfo, ...] = (),
        out_info: OutInfo | None = None,
        text: str = "HloModule compiled_module",
    ) -> None:
        """Initialize ``Compiled``.

        Args:
            fun: Executable callable.
            args_info: Tuple of ``ArgInfo`` metadata.
            out_info: Output ``OutInfo`` metadata.
            text: Textual representation of the compiled program.
        """
        self._fun = fun
        self.args_info = args_info
        self.out_info = out_info if out_info is not None else OutInfo()
        self._text = text

    def as_text(self) -> str:
        """Return human-readable text of the compiled executable.

        Returns:
            str: Textual representation.
        """
        return self._text

    def cost_analysis(self) -> dict[str, float]:
        """Return estimated FLOP and memory cost metrics.

        Returns:
            dict[str, float]: Cost analysis dictionary.
        """
        return {"flops": 0.0, "bytes accessed": 0.0}

    def memory_analysis(self) -> dict[str, int]:
        """Return estimated memory usage metrics in bytes.

        Returns:
            dict[str, int]: Memory usage dictionary.
        """
        return {"temp_size_in_bytes": 0, "output_size_in_bytes": 0}

    def __call__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute the compiled function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Execution result.
        """
        if self._fun is not None:
            return self._fun(*args, **kwargs)
        return None


class Lowered:
    """Lowering of a traced function prior to backend compilation."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
        args_info: tuple[ArgInfo, ...] = (),
        out_info: OutInfo | None = None,
        ir_text: str = "module @jit_lowered {}",
    ) -> None:
        """Initialize ``Lowered``.

        Args:
            fun: Underlying callable.
            args_info: Argument info descriptors.
            out_info: Output info descriptor.
            ir_text: Lowered StableHLO/IR text representation.
        """
        self._fun = fun
        self.args_info = args_info
        self.out_info = out_info if out_info is not None else OutInfo()
        self._ir_text = ir_text

    def as_text(self, dialect: str | None = None) -> str:
        """Return textual IR representation of the lowered computation.

        Args:
            dialect: Optional IR dialect name.

        Returns:
            str: Textual IR string.
        """
        return self._ir_text

    def compile(self, compiler_options: CompilerOptions | None = None) -> Compiled:
        """Compile this lowered computation into a :class:`Compiled` executable.

        Args:
            compiler_options: Optional compiler flags.

        Returns:
            Compiled: Compiled executable stage.
        """
        return Compiled(fun=self._fun, args_info=self.args_info, out_info=self.out_info, text=self._ir_text)

    def cost_analysis(self) -> dict[str, float]:
        """Return pre-compilation cost analysis estimates.

        Returns:
            dict[str, float]: Cost metrics.
        """
        return {"flops": 0.0}


class Traced:
    """Traced representation of a function specialized to abstract argument shapes."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
        args_info: tuple[ArgInfo, ...] = (),
        out_info: OutInfo | None = None,
    ) -> None:
        """Initialize ``Traced``.

        Args:
            fun: Traced callable.
            args_info: Argument metadata tuple.
            out_info: Output metadata descriptor.
        """
        self._fun = fun
        self.args_info = args_info
        self.out_info = out_info if out_info is not None else OutInfo()

    def lower(self, *, lowering_platforms: tuple[str, ...] | None = None) -> Lowered:
        """Lower the traced representation to IR.

        Args:
            lowering_platforms: Optional target platform names.

        Returns:
            Lowered: The lowered stage object.
        """
        return Lowered(fun=self._fun, args_info=self.args_info, out_info=self.out_info)


class Wrapped:
    """A function wrapped by a transformation (such as ``jit``) ready to be traced, lowered, and compiled."""

    def __init__(
        self,
        fun: VariadicCallable[ArrayLikeSpec] | None = None,
    ) -> None:
        """Initialize ``Wrapped``.

        Args:
            fun: The wrapped Python callable.
        """
        self._fun = fun

    def trace(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> Traced:
        """Trace the wrapped function for the given sample arguments.

        Args:
            *args: Sample positional arguments.
            **kwargs: Sample keyword arguments.

        Returns:
            Traced: Traced stage representation.
        """
        return Traced(fun=self._fun)

    def lower(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> Lowered:
        """Trace and lower the wrapped function for the given sample arguments.

        Args:
            *args: Sample positional arguments.
            **kwargs: Sample keyword arguments.

        Returns:
            Lowered: Lowered stage representation.
        """
        return self.trace(*args, **kwargs).lower()

    def __call__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the wrapped callable directly.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Function output.
        """
        if self._fun is not None:
            return self._fun(*args, **kwargs)
        return None


__all__ = [
    "ArgInfo",
    "Compiled",
    "CompilerOptions",
    "Lowered",
    "OutInfo",
    "Traced",
    "Wrapped",
]
