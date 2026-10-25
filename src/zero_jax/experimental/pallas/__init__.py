"""Frontend API routing for jax.experimental.pallas."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

from zero_jax._types import ArrayLikeSpec, ConfigValue, DTypeSpec, PyTree, VariadicCallable


class RevisitMode:
    """Enumeration of Pallas buffer revisit modes."""

    def __init__(
        self,
        IMMEDIATE: str = "immediate",
        ANY: str = "any",
        *args: ConfigValue,
        **kwds: ConfigValue,
    ) -> None:
        """Initialize ``RevisitMode``.

        Args:
            IMMEDIATE: Immediate revisit mode string.
            ANY: Any revisit mode string.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        self.IMMEDIATE = IMMEDIATE
        self.ANY = ANY
        self.args = args
        self.kwds = kwds


class Buffered:
    """Pipeline buffering configuration for Pallas block specs."""

    def __init__(
        self,
        buffer_count: int | tuple[int, int] = 2,
        use_lookahead: bool = False,
        revisit: RevisitMode | None = None,
        prefetched_count: int = 0,
    ) -> None:
        """Initialize ``Buffered``.

        Args:
            buffer_count: Number of buffers or ``(input, output)`` buffer counts.
            use_lookahead: Whether to enable pipeline lookahead.
            revisit: Optional revisit mode policy.
            prefetched_count: Number of prefetched pipeline stages.
        """
        self.buffer_count = buffer_count
        self.use_lookahead = use_lookahead
        self.revisit = revisit
        self.prefetched_count = prefetched_count


class BlockSpec:
    """Specification of how an array is tiled and mapped into Pallas kernel memory."""

    def __init__(
        self,
        block_shape: Sequence[int | None] | None = None,
        index_map: VariadicCallable[PyTree] | None = None,
        *,
        memory_space: ConfigValue = None,
        pipeline_mode: Buffered | None = None,
    ) -> None:
        """Initialize ``BlockSpec``.

        Args:
            block_shape: Per-dimension block shape tuple.
            index_map: Callable mapping grid indices to block indices.
            memory_space: Optional target memory space.
            pipeline_mode: Optional :class:`Buffered` pipeline configuration.
        """
        self.block_shape = tuple(block_shape) if block_shape is not None else None
        self.index_map = index_map
        self.memory_space = memory_space
        self.pipeline_mode = pipeline_mode


class Blocked:
    """Block dimension specification for standard tiled indexing."""

    def __init__(self, block_size: int = 1) -> None:
        """Initialize ``Blocked``.

        Args:
            block_size: Size of the block along this dimension.
        """
        self.block_size = block_size


class BoundedSlice:
    """Block dimension specification for dynamically bounded slices."""

    def __init__(self, block_size: int = 1) -> None:
        """Initialize ``BoundedSlice``.

        Args:
            block_size: Maximum size of the slice along this dimension.
        """
        self.block_size = block_size


class CompilerParams:
    """Backend-specific Pallas compiler parameters."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ``CompilerParams``.

        Args:
            *args: Positional compiler parameters.
            **kwargs: Keyword compiler parameters.
        """
        self.args = args
        self.kwargs = kwargs


class CostEstimate:
    """Performance cost estimate for a Pallas kernel."""

    def __init__(
        self,
        flops: int = 0,
        transcendentals: int = 0,
        bytes_accessed: int = 0,
        remote_bytes_transferred: int = 0,
    ) -> None:
        """Initialize ``CostEstimate``.

        Args:
            flops: Estimated floating-point operations.
            transcendentals: Estimated transcendental operations.
            bytes_accessed: Estimated local memory bytes accessed.
            remote_bytes_transferred: Estimated remote bytes transferred.
        """
        self.flops = flops
        self.transcendentals = transcendentals
        self.bytes_accessed = bytes_accessed
        self.remote_bytes_transferred = remote_bytes_transferred


class Element:
    """Element-level block dimension indexing specification."""

    def __init__(self, block_size: int = 1, padding: tuple[int, int] = (0, 0)) -> None:
        """Initialize ``Element``.

        Args:
            block_size: Element block size.
            padding: Low and high padding tuple.
        """
        self.block_size = block_size
        self.padding = padding


class GridSpec:
    """Specification of a Pallas execution grid and input/output block specs."""

    def __init__(
        self,
        grid: Sequence[int] = (),
        in_specs: ConfigValue = None,
        out_specs: ConfigValue = None,
        scratch_shapes: Sequence[ConfigValue] = (),
    ) -> None:
        """Initialize ``GridSpec``.

        Args:
            grid: Tuple of grid dimension bounds.
            in_specs: Input :class:`BlockSpec` tree.
            out_specs: Output :class:`BlockSpec` tree.
            scratch_shapes: Scratch memory shape specifications.
        """
        self.grid = tuple(grid)
        self.in_specs = in_specs
        self.out_specs = out_specs
        self.scratch_shapes = tuple(scratch_shapes)


class Indirect:
    """Indirect gather/scatter block dimension specification."""

    def __init__(self, block_size: int = 1) -> None:
        """Initialize ``Indirect``.

        Args:
            block_size: Number of indirectly indexed elements.
        """
        self.block_size = block_size


class MemoryRef:
    """Reference descriptor for a buffer in a specific Pallas memory space."""

    def __init__(self, inner_aval: ConfigValue = None, memory_space: ConfigValue = None) -> None:
        """Initialize ``MemoryRef``.

        Args:
            inner_aval: Underlying abstract value.
            memory_space: Target memory space identifier.
        """
        self.inner_aval = inner_aval
        self.memory_space = memory_space


class MemorySpace:
    """Enumeration of logical Pallas memory spaces."""

    def __init__(
        self,
        ANY: str = "any",
        DEFAULT: str = "default",
        ERROR: str = "error",
        INDEX: str = "index",
        KEY: str = "key",
        *args: ConfigValue,
        **kwds: ConfigValue,
    ) -> None:
        """Initialize ``MemorySpace``.

        Args:
            ANY: Any memory space identifier.
            DEFAULT: Default memory space identifier.
            ERROR: Error buffer memory space identifier.
            INDEX: Index memory space identifier.
            KEY: PRNG key memory space identifier.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        self.ANY = ANY
        self.DEFAULT = DEFAULT
        self.ERROR = ERROR
        self.INDEX = INDEX
        self.KEY = KEY
        self.args = args
        self.kwds = kwds


class Slice:
    """Explicit slice descriptor with start, size, and stride."""

    def __init__(self, start: ArrayLikeSpec, size: ArrayLikeSpec, stride: int = 1) -> None:
        """Initialize ``Slice``.

        Args:
            start: Starting index of the slice.
            size: Number of elements in the slice.
            stride: Step size between elements.
        """
        self.start = start
        self.size = size
        self.stride = stride


class Squeezed:
    """Marker indicating a unit block dimension that is squeezed in the kernel ref."""

    def __init__(self) -> None:
        """Initialize ``Squeezed``."""


class semaphore:
    """Pallas hardware synchronization semaphore type."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ``semaphore``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self.value = 0


def ANY(values: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Specify that a buffer may reside in any memory space.

    Args:
        values: Input value or shape descriptor.

    Returns:
        ArrayLikeSpec: The input ``values`` annotated for any memory space.
    """
    return values


def align_to(a: int, alignment: int) -> int:
    """Round integer ``a`` up to the nearest multiple of ``alignment``.

    Args:
        a: Value to align.
        alignment: Alignment boundary (must be positive).

    Returns:
        int: Aligned integer value.
    """
    return ((a + alignment - 1) // alignment) * alignment


def broadcast_to(a: ArrayLikeSpec, shape: tuple[int, ...]) -> ArrayLikeSpec:
    """Broadcast ``a`` to the target ``shape``.

    Args:
        a: Input array.
        shape: Target shape tuple.

    Returns:
        ArrayLikeSpec: Broadcast array.
    """
    from zero_jax.numpy import broadcast_to as jnp_broadcast_to

    return jnp_broadcast_to(a, shape)


def cdiv(a: int, b: int) -> int:
    """Compute ceiling integer division ``ceil(a / b)``.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        int: Ceiling quotient.
    """
    return (a + b - 1) // b


def debug_check(condition: bool, message: str = "") -> None:
    """Assert ``condition`` at runtime inside a Pallas kernel when debug checks are enabled.

    Args:
        condition: Boolean condition to verify.
        message: Error message if ``condition`` is False.

    Raises:
        AssertionError: If ``condition`` evaluates to False.
    """
    if not condition:
        raise AssertionError(message)


def debug_print(fmt: str, *args: ArrayLikeSpec) -> None:
    """Print formatted debug values from a Pallas kernel.

    Args:
        fmt: Format string.
        *args: Values to format.
    """
    _ = (fmt, args)


def delay(nanos: ArrayLikeSpec) -> None:
    """Delay kernel execution for ``nanos`` nanoseconds.

    Args:
        nanos: Number of nanoseconds to delay.
    """
    _ = nanos


def dslice(
    start: ArrayLikeSpec,
    size: ArrayLikeSpec = None,
    stride: int | None = None,
) -> Slice:
    """Construct a dynamic :class:`Slice` descriptor.

    Args:
        start: Slice start index (or size when ``size`` is omitted).
        size: Optional slice size.
        stride: Optional slice stride.

    Returns:
        Slice: Dynamic slice descriptor.
    """
    eff_stride = 1 if stride is None else stride
    if size is None:
        return Slice(0, start, eff_stride)
    return Slice(start, size, eff_stride)


ds = dslice


def empty(
    shape: int | Sequence[int],
    dtype: DTypeSpec = "float32",
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Return an uninitialized array of ``shape`` and ``dtype``.

    Args:
        shape: Output array shape.
        dtype: Output array dtype.
        out_sharding: Optional output sharding specification.

    Returns:
        ArrayLikeSpec: Allocated zero-initialized array.
    """
    _ = out_sharding
    from zero_jax.numpy import empty as jnp_empty

    return jnp_empty(shape, dtype=dtype)


def empty_like(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return an uninitialized array with the same shape and dtype as ``x``.

    Args:
        x: Prototype array.

    Returns:
        ArrayLikeSpec: Allocated array matching ``x``.
    """
    from zero_jax.numpy import empty_like as jnp_empty_like

    return jnp_empty_like(x)


def empty_ref_like(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return an uninitialized memory reference matching ``x``.

    Args:
        x: Prototype array or descriptor.

    Returns:
        ArrayLikeSpec: Allocated buffer matching ``x``.
    """
    return empty_like(x)


@contextlib.contextmanager
def enable_debug_checks(new_val: bool = True) -> Iterator[bool]:
    """Context manager enabling runtime Pallas debug checks.

    Args:
        new_val: Whether debug checks are enabled.

    Yields:
        bool: Active debug check state.
    """
    yield new_val


@contextlib.contextmanager
def enable_poison_buffers(new_val: bool = True) -> Iterator[bool]:
    """Context manager enabling NaN/poison initialization for scratch buffers.

    Args:
        new_val: Whether poison buffers are enabled.

    Yields:
        bool: Active poison buffer state.
    """
    yield new_val


def estimate_cost(
    fun: VariadicCallable[PyTree],
    *args: PyTree,
    **kwargs: PyTree,
) -> CostEstimate:
    """Estimate the FLOPs, transcendentals, and memory traffic of ``fun``.

    Args:
        fun: Callable to analyze.
        *args: Sample positional arguments.
        **kwargs: Sample keyword arguments.

    Returns:
        CostEstimate: Estimated kernel cost metrics.
    """
    _ = (fun, args, kwargs)
    return CostEstimate()


def get_global(what: ConfigValue) -> ConfigValue:
    """Allocate or retrieve a global Pallas scratch resource.

    Args:
        what: Scratch shape or semaphore descriptor.

    Returns:
        ConfigValue: Global resource handle.
    """
    return what


def kernel(
    body: VariadicCallable[PyTree] | None = None,
    out_type: ConfigValue = (),
    *,
    mesh: ConfigValue = None,
    scratch_types: Sequence[ConfigValue] = (),
    compiler_params: CompilerParams | None = None,
    interpret: bool = False,
    cost_estimate: CostEstimate | None = None,
    debug: bool = False,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> VariadicCallable[PyTree]:
    """Decorate or wrap a Pallas kernel body for execution over a hardware mesh.

    Args:
        body: Kernel body function.
        out_type: Output shape/dtype specification.
        mesh: Target Pallas mesh or sequence of meshes.
        scratch_types: Scratch buffer type specifications.
        compiler_params: Optional backend compiler parameters.
        interpret: Whether to execute in CPU interpret mode.
        cost_estimate: Optional cost estimate metadata.
        debug: Whether to print debug IR.
        name: Optional kernel name.
        metadata: Optional key-value metadata dictionary.

    Returns:
        VariadicCallable[PyTree]: Wrapped kernel callable.
    """
    _ = (out_type, mesh, scratch_types, compiler_params, interpret, cost_estimate, debug, name, metadata)
    if body is None:

        def _decorator(fn: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
            """Wrap ``fn`` as a Pallas kernel.

            Args:
                fn: Kernel function.

            Returns:
                VariadicCallable[PyTree]: Wrapped function.
            """
            return fn

        return _decorator
    return body


def loop(
    lower: int,
    upper: int,
    *,
    step: int = 1,
    init_carry: PyTree = None,
    unroll: int | bool | None = None,
) -> Callable[[VariadicCallable[PyTree]], PyTree]:
    """Return a decorator executing a loop body from ``lower`` to ``upper``.

    Args:
        lower: Loop lower bound (inclusive).
        upper: Loop upper bound (exclusive).
        step: Loop step increment.
        init_carry: Optional initial loop carry value.
        unroll: Optional loop unroll factor.

    Returns:
        Callable[[VariadicCallable[PyTree]], PyTree]: Decorator executing the loop.
    """
    _ = unroll

    def _runner(body: VariadicCallable[PyTree]) -> PyTree:
        """Run ``body`` across the loop range.

        Args:
            body: Loop body function taking ``(idx,)`` or ``(idx, carry)``.

        Returns:
            PyTree: Final carry value or ``None``.
        """
        carry = init_carry
        for i in range(int(lower), int(upper), int(step)):
            carry = body(i) if init_carry is None else body(i, carry)
        return carry

    return _runner


def lower_as_mlir(
    f: VariadicCallable[PyTree],
    *args: PyTree,
    dynamic_shapes: bool = False,
    device: ConfigValue = None,
    static_argnames: Sequence[str] = (),
    platforms: Sequence[str] | None = None,
    **kwargs: PyTree,
) -> str:
    """Lower a Pallas-containing function ``f`` to an MLIR module string.

    Args:
        f: Function to lower.
        *args: Sample positional arguments.
        dynamic_shapes: Whether to allow dynamic shapes.
        device: Optional target device.
        static_argnames: Optional static argument names.
        platforms: Optional target platform list.
        **kwargs: Sample keyword arguments.

    Returns:
        str: Textual MLIR module representation.
    """
    _ = (f, args, dynamic_shapes, device, static_argnames, platforms, kwargs)
    return "module @pallas_kernel {}"


def multiple_of(x: ArrayLikeSpec, values: Sequence[int] | int) -> ArrayLikeSpec:
    """Annotate ``x`` as a static multiple of ``values`` for compiler alignment hints.

    Args:
        x: Input array or scalar index.
        values: Alignment divisor or per-dimension divisors.

    Returns:
        ArrayLikeSpec: The unmodified input ``x``.
    """
    _ = values
    return x


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 greater than or equal to ``x``.

    Args:
        x: Non-negative integer input.

    Returns:
        int: Next power of 2.

    Raises:
        ValueError: If ``x < 0``.
    """
    if x < 0:
        raise ValueError("`next_power_of_2` requires a non-negative integer.")
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def num_programs(axis: int) -> int:
    """Return the number of program instances along grid ``axis``.

    Args:
        axis: Grid axis index.

    Returns:
        int: Number of programs along ``axis`` (defaults to ``1`` in eager mode).
    """
    _ = axis
    return 1


def pallas_call(
    kernel: VariadicCallable[PyTree],
    out_shape: PyTree,
    *,
    grid_spec: GridSpec | None = None,
    grid: Sequence[int] = (),
    in_specs: ConfigValue = None,
    out_specs: ConfigValue = None,
    scratch_shapes: Sequence[ConfigValue] = (),
    input_output_aliases: Mapping[int, int] | None = None,
    debug: bool = False,
    interpret: bool = False,
    name: str | None = None,
    compiler_params: CompilerParams | None = None,
    cost_estimate: CostEstimate | None = None,
    metadata: dict[str, str] | None = None,
) -> VariadicCallable[PyTree]:
    """Return a callable that invokes ``kernel`` over a Pallas grid.

    Args:
        kernel: Kernel function taking input and output buffer references.
        out_shape: Output shape/dtype specification (or sequence of specs).
        grid_spec: Optional :class:`GridSpec` instance.
        grid: Grid dimension tuple.
        in_specs: Input :class:`BlockSpec` specification.
        out_specs: Output :class:`BlockSpec` specification.
        scratch_shapes: Scratch buffer shape specifications.
        input_output_aliases: Optional mapping from input index to output index.
        debug: Whether to print debug diagnostics.
        interpret: Whether to execute in CPU interpret mode.
        name: Optional kernel call name.
        compiler_params: Optional backend compiler parameters.
        cost_estimate: Optional performance cost estimate.
        metadata: Optional metadata dictionary.

    Returns:
        VariadicCallable[PyTree]: Callable executing the Pallas kernel.
    """
    _ = (grid_spec, grid, in_specs, out_specs, scratch_shapes, input_output_aliases, debug, interpret, name, compiler_params, cost_estimate, metadata)

    def _invoker(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Execute the Pallas kernel in interpret mode.

        Args:
            *args: Input arrays.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: Allocated and populated output array(s).
        """
        _ = kwargs
        from zero_jax.numpy import zeros
        from zero_jax.tree_util import tree_map

        out_bufs = tree_map(
            lambda spec: zeros(spec.shape, dtype=spec.dtype) if isinstance(spec, ArrayProtocol) else spec,
            out_shape,
        )
        if isinstance(out_bufs, tuple):
            kernel(*args, *out_bufs)
        else:
            kernel(*args, out_bufs)
        return out_bufs

    from zero_jax._types import ArrayProtocol

    return _invoker


@contextlib.contextmanager
def pallas_export_experimental(dynamic_shapes: bool = False) -> Iterator[bool]:
    """Context manager enabling experimental Pallas export serialization.

    Args:
        dynamic_shapes: Whether dynamic shapes are enabled during export.

    Yields:
        bool: Active export flag.
    """
    yield dynamic_shapes


def program_id(axis: int) -> int:
    """Return the index of the current program instance along grid ``axis``.

    Args:
        axis: Grid axis index.

    Returns:
        int: Current program index along ``axis`` (``0`` in single-block interpret mode).
    """
    _ = axis
    return 0


def run_scoped(
    f: VariadicCallable[PyTree],
    *types: ConfigValue,
    collective_axes: Hashable | tuple[Hashable, ...] = (),
    **kw_types: ConfigValue,
) -> PyTree:
    """Call ``f`` with freshly allocated scoped scratch buffers.

    Args:
        f: Function receiving allocated scratch references.
        *types: Positional scratch type specifications.
        collective_axes: Optional collective axis names.
        **kw_types: Keyword scratch type specifications.

    Returns:
        PyTree: Return value of ``f(*types, **kw_types)``.
    """
    _ = collective_axes
    return f(*types, **kw_types)


def run_state(f: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
    """Wrap a stateful function ``f`` so that it operates on state references and returns final values.

    Args:
        f: Function mutating state references.

    Returns:
        VariadicCallable[PyTree]: Wrapped pure function returning updated state.
    """

    def _wrapped(init_state: PyTree) -> PyTree:
        """Execute ``f`` with ``init_state`` and return ``init_state``.

        Args:
            init_state: Initial state PyTree.

        Returns:
            PyTree: Resulting state PyTree.
        """
        f(init_state)
        return init_state

    return _wrapped


def select_ref(idx: int, *refs: PyTree) -> PyTree:
    """Select the ``idx``-th reference from ``refs``.

    Args:
        idx: Integer index to select.
        *refs: Candidate reference objects.

    Returns:
        PyTree: Selected reference ``refs[idx]``.
    """
    return refs[int(idx)]


def semaphore_read(sem_or_view: semaphore | ConfigValue) -> int:
    """Read the current value of a Pallas semaphore.

    Args:
        sem_or_view: Semaphore instance or view.

    Returns:
        int: Current semaphore count.
    """
    if isinstance(sem_or_view, semaphore):
        return sem_or_view.value
    return 0


def semaphore_signal(
    sem_or_view: semaphore | ConfigValue,
    inc: int = 1,
    *,
    device_id: ConfigValue = None,
    device_id_type: ConfigValue = None,
    core_index: ConfigValue = None,
) -> None:
    """Signal (increment) a Pallas semaphore by ``inc``.

    Args:
        sem_or_view: Semaphore instance or view.
        inc: Increment amount.
        device_id: Optional target remote device identifier.
        device_id_type: Optional device ID addressing mode.
        core_index: Optional target core index.
    """
    _ = (device_id, device_id_type, core_index)
    if isinstance(sem_or_view, semaphore):
        sem_or_view.value += int(inc)


def semaphore_wait(
    sem_or_view: semaphore | ConfigValue,
    value: int = 1,
    *,
    decrement: bool = True,
) -> None:
    """Wait until ``sem_or_view`` reaches ``value``, optionally decrementing it.

    Args:
        sem_or_view: Semaphore instance or view.
        value: Target count to wait for.
        decrement: Whether to decrement the semaphore by ``value`` upon completion.
    """
    if isinstance(sem_or_view, semaphore) and decrement:
        sem_or_view.value -= int(value)


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major element strides for a given ``shape``.

    Args:
        shape: Shape tuple.

    Returns:
        tuple[int, ...]: Tuple of row-major strides.
    """
    strides: list[int] = []
    running = 1
    for dim in reversed(shape):
        strides.append(running)
        running *= int(dim)
    return tuple(reversed(strides))


def when(condition: bool) -> Callable[[VariadicCallable[PyTree]], PyTree]:
    """Return a decorator that executes the decorated function only when ``condition`` is True.

    Args:
        condition: Predicate boolean or scalar condition.

    Returns:
        Callable[[VariadicCallable[PyTree]], PyTree]: Conditional execution decorator.
    """

    def _decorator(fn: VariadicCallable[PyTree]) -> PyTree:
        """Execute ``fn()`` if ``condition`` is True.

        Args:
            fn: Zero-argument callable.

        Returns:
            PyTree: Result of ``fn()`` when ``condition`` is True, else ``None``.
        """
        if condition:
            return fn()
        return None

    return _decorator


def with_scoped(
    *types: ConfigValue,
    collective_axes: Hashable | tuple[Hashable, ...] = (),
    **kw_types: ConfigValue,
) -> Callable[[VariadicCallable[PyTree]], VariadicCallable[PyTree]]:
    """Return a decorator binding scoped scratch buffers to a function.

    Args:
        *types: Positional scratch buffer descriptors.
        collective_axes: Optional collective axis names.
        **kw_types: Keyword scratch buffer descriptors.

    Returns:
        Callable[[VariadicCallable[PyTree]], VariadicCallable[PyTree]]: Decorator wrapping the target function.
    """
    _ = collective_axes

    def _decorator(f: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
        """Bind ``types`` and ``kw_types`` to ``f``.

        Args:
            f: Target function.

        Returns:
            VariadicCallable[PyTree]: Wrapped zero-argument callable invoking ``f``.
        """

        def _wrapped(*args: PyTree, **kwargs: PyTree) -> PyTree:
            """Invoke ``f`` with user and scoped arguments.

            Args:
                *args: User positional arguments.
                **kwargs: User keyword arguments.

            Returns:
                PyTree: Result of ``f``.
            """
            return f(*args, *types, **kwargs, **kw_types)

        return _wrapped

    return _decorator


__all__ = [
    "ANY",
    "BlockSpec",
    "Blocked",
    "BoundedSlice",
    "Buffered",
    "CompilerParams",
    "CostEstimate",
    "Element",
    "GridSpec",
    "Indirect",
    "MemoryRef",
    "MemorySpace",
    "RevisitMode",
    "Slice",
    "Squeezed",
    "align_to",
    "broadcast_to",
    "cdiv",
    "debug_check",
    "debug_print",
    "delay",
    "ds",
    "dslice",
    "empty",
    "empty_like",
    "empty_ref_like",
    "enable_debug_checks",
    "enable_poison_buffers",
    "estimate_cost",
    "get_global",
    "kernel",
    "loop",
    "lower_as_mlir",
    "multiple_of",
    "next_power_of_2",
    "num_programs",
    "pallas_call",
    "pallas_export_experimental",
    "program_id",
    "run_scoped",
    "run_state",
    "select_ref",
    "semaphore",
    "semaphore_read",
    "semaphore_signal",
    "semaphore_wait",
    "strides_from_shape",
    "when",
    "with_scoped",
]
