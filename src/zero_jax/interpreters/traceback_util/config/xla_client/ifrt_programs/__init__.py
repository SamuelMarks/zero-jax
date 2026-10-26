"""Frontend API routing for ifrt_programs."""

from __future__ import annotations


class CompileOptions:
    """IFRT program compilation options."""

    def __init__(self, num_replicas: int = 1, num_partitions: int = 1) -> None:
        """Initialize ``CompileOptions``.

        Args:
            num_replicas: Number of computation replicas.
            num_partitions: Number of model partitions.
        """
        self.num_replicas: int = int(num_replicas)
        self.num_partitions: int = int(num_partitions)


class Program:
    """IFRT executable program container."""

    def __init__(self, code: str | bytes = b"", kind: str = "hlo") -> None:
        """Initialize ``Program``.

        Args:
            code: Serialized HLO or plugin program payload.
            kind: Program format identifier (``"hlo"`` or ``"plugin"``).
        """
        self.code: bytes = code.encode("utf-8") if isinstance(code, str) else bytes(code)
        self.kind: str = str(kind)


def make_hlo_program(mlir_module: str | bytes = b"") -> Program:
    """Create an IFRT HLO :class:`Program` from ``mlir_module``.

    Args:
        mlir_module: Serialized MLIR/StableHLO module bytes or text.

    Returns:
        Program: Constructed HLO IFRT program.
    """
    return Program(code=mlir_module, kind="hlo")


def make_plugin_compile_options() -> CompileOptions:
    """Create default plugin :class:`CompileOptions`.

    Returns:
        CompileOptions: Default IFRT plugin compile options.
    """
    return CompileOptions()


def make_plugin_program(data: str | bytes = b"") -> Program:
    """Create an IFRT plugin :class:`Program` from ``data``.

    Args:
        data: Serialized plugin program payload.

    Returns:
        Program: Constructed plugin IFRT program.
    """
    return Program(code=data, kind="plugin")


def make_xla_compile_options(options: CompileOptions | None = None) -> CompileOptions:
    """Wrap XLA :class:`CompileOptions` for IFRT compilation.

    Args:
        options: Existing :class:`CompileOptions` or ``None``.

    Returns:
        CompileOptions: IFRT XLA compile options.
    """
    return options if options is not None else CompileOptions()


__all__ = [
    "CompileOptions",
    "Program",
    "make_hlo_program",
    "make_plugin_compile_options",
    "make_plugin_program",
    "make_xla_compile_options",
]
