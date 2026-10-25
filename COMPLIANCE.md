# API Compliance Declaration

This framework is a frontend routing wrapper over the `ml-switcheroo-compiler` backend engine and `ml-switcheroo-ir` Intermediate Representation, grounded against `ml-ecosystem-snapshots` (`v0.0.5`).
All mathematical and gradient computations evaluate through the central abstract IR.

- **100% Target API Parity**: Maintained and verified against `ml-ecosystem-snapshots` (`v0.0.5`).
- **100% Documentation Coverage**: Maintained across all files, modules, classes, functions, methods, and arguments (`interrogate -f 100` and `ruff` Google `D100`–`D417`).
- **100% Test Coverage**: Maintained across 100% of functions, lines, and branches (`--cov-branch --cov-fail-under=100`), evaluating the compiler backend with zero `# pragma: no cover` suppressions.
- **Strong Typing**: Maintained with explicit type annotations everywhere; zero `Any`, zero `object` annotations, zero `typing.cast`, and zero dynamic `getattr`/`setattr` or module-level `__getattr__` dunder tricks.
- **No 3rd Party Executable Dependencies**: Maintained (purely defers to `ml-switcheroo-compiler`, `ml-switcheroo-ir`, `pydantic`, and standard library types).
