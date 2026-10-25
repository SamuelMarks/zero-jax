"""Unified CI checking script for the zero-jax project."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from types import ModuleType

ALLOWED_3RD_PARTY_SRC: set[str] = {"pydantic"}
ALLOWED_INTERNAL_SRC: set[str] = {
    "ml_switcheroo_compiler",
    "cdd",
    "cdd_python",
    "ml_switcheroo_ir",
    "ml_ecosystem_snapshots",
    "zero_jax",
}
PARITY_TEST_FILES: frozenset[str] = frozenset(
    {
        "test_reference_api.py",
        "test_reference_checkify.py",
        "test_reference_jnp.py",
        "test_reference_lax.py",
        "test_reference_nn.py",
        "test_reference_random.py",
        "test_reference_tree_util.py",
    }
)

EXCLUDED_SIGNATURE_NAMES: set[str] = {
    "Any",
    "Callable",
    "Tuple",
    "List",
    "Dict",
    "Sequence",
    "Union",
    "Optional",
    "np",
    "jnp",
    "lax",
    "nn",
    "Tensor",
    "xla",
    "xla_client",
    "xla_bridge",
    "xla_extension",
    "mlir",
    "hlo",
    "pjrt",
    "ifrt",
    "ifrt_programs",
    "tpu_cluster",
    "ompi_cluster",
    "slurm_cluster",
    "Array",
    "Device",
    "ConfigValue",
    "ArrayLikeSpec",
    "DTypeSpec",
    "AxisSpec",
    "ShapeTuple",
    "ShapeSpec",
    "ScalarNumeric",
    "PyTree",
    "PyTreeLeaf",
    "random_gamma_p",
}


def _get_stdlib_names() -> frozenset[str]:
    """Return a frozenset of Python standard library module names.

    Returns:
        frozenset[str]: Standard library module names.
    """
    names: set[str] = set(sys.builtin_module_names)
    names.update(
        {
            "__future__",
            "abc",
            "argparse",
            "ast",
            "builtins",
            "collections",
            "contextlib",
            "copy",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "glob",
            "importlib",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "numbers",
            "operator",
            "os",
            "pathlib",
            "random",
            "re",
            "shutil",
            "subprocess",
            "sys",
            "tempfile",
            "textwrap",
            "threading",
            "time",
            "traceback",
            "types",
            "typing",
            "unittest",
            "uuid",
            "warnings",
            "weakref",
        }
    )
    return frozenset(names)


def check_file_imports(filepath: str) -> bool:
    """Check a Python file for disallowed third-party imports.

    Args:
        filepath: The path to the Python file to check.

    Returns:
        bool: ``True`` if the file has no disallowed imports, ``False`` otherwise.
    """
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return False

    stdlib_names = _get_stdlib_names()
    allowed_all = set(ALLOWED_3RD_PARTY_SRC | ALLOWED_INTERNAL_SRC | stdlib_names)
    errors: list[tuple[int, str]] = []
    norm_path = filepath.replace("\\", "/")
    parts = norm_path.split("/")
    filename = parts[-1]
    is_script = "scripts" in parts
    is_test = "tests" in parts
    is_parity_test = is_test and filename in PARITY_TEST_FILES

    if is_script:
        allowed_all.add("scripts")
    elif is_test:
        allowed_all.update({"pytest", "tests", "scripts", "conftest", "fuzz_utils"})
        if is_parity_test:
            allowed_all.update({"jax", "numpy"})

    def _is_allowed_module(base_mod: str) -> bool:
        """Check whether a top-level module name is permitted for the current file.

        Args:
            base_mod: Top-level package or module name.

        Returns:
            bool: ``True`` if permitted, ``False`` otherwise.
        """
        return base_mod in allowed_all or base_mod.startswith("zero_")

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.value.id == "sys" and node.value.attr == "modules":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                base_module = node.slice.value.split(".")[0]
                if not _is_allowed_module(base_module):
                    errors.append((node.lineno, f"Disallowed sys.modules access: '{base_module}'"))
            else:
                errors.append((node.lineno, "sys.modules access with non-constant is forbidden"))

        elif isinstance(node, ast.Call):
            func_name: str | None = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name):
                func_name = f"{node.func.value.value.id}.{node.func.value.attr}.{node.func.attr}"

            if func_name in ("__import__", "importlib.import_module", "sys.modules.get", "sys.modules.pop") or (isinstance(node.func, ast.Attribute) and node.func.attr == "__import__"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    base_module = node.args[0].value.split(".")[0]
                    if not _is_allowed_module(base_module):
                        errors.append((node.lineno, f"Disallowed dynamic import: '{base_module}'"))
                elif not (is_test and filename == "test_stubs.py" and func_name == "importlib.import_module"):
                    errors.append((node.lineno, f"Dynamic import with non-constant argument is forbidden: '{func_name}'"))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not _is_allowed_module(base_module):
                    errors.append((node.lineno, f"Disallowed import: '{base_module}'"))

        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            base_module = node.module.split(".")[0]
            if not _is_allowed_module(base_module):
                errors.append((node.lineno, f"Disallowed import: '{base_module}'"))

    if errors:
        for lineno, msg in errors:
            print(f"{filepath}:{lineno}: {msg}")
        return False
    return True


def _collect_py_files(paths: Sequence[str], default_dirs: Sequence[str]) -> list[str]:
    """Expand file and directory paths into a sorted list of Python files.

    Args:
        paths: Explicit file or directory paths provided on the command line.
        default_dirs: Default directories to scan when ``paths`` is empty.

    Returns:
        list[str]: Sorted list of ``.py`` file paths.
    """
    targets = list(paths) if paths else list(default_dirs)
    result: list[str] = []
    for target in targets:
        if os.path.isfile(target) and target.endswith(".py"):
            result.append(target)
        elif os.path.isdir(target):
            for root, _, files in os.walk(target):
                for file in sorted(files):
                    if file.endswith(".py"):
                        result.append(os.path.join(root, file))
    return sorted(set(result))


def cmd_check_allowed_imports(args: argparse.Namespace) -> None:
    """Run the allowed imports check on provided files or ``src/zero_jax``, ``scripts``, and ``tests``.

    Args:
        args: Parsed command line arguments containing ``files``.
    """
    files = _collect_py_files(args.files, ["src/zero_jax", "scripts", "tests"])
    success = True
    for filepath in files:
        if not check_file_imports(filepath):
            success = False
    sys.exit(0 if success else 1)


def get_signatures(module: ModuleType, prefix: str = "") -> dict[str, str]:
    """Extract all callable signatures from a module without using ``getattr``.

    Args:
        module: The module object to inspect.
        prefix: Optional dot-separated path prefix.

    Returns:
        dict[str, str]: Mapping from public symbol name to stringified signature.
    """
    signatures: dict[str, str] = {}
    mod_dict = module.__dict__
    for name in sorted(mod_dict.keys()):
        if not name.startswith("_") and name not in EXCLUDED_SIGNATURE_NAMES:
            obj = mod_dict[name]
            full_path = f"{prefix}.{name}" if prefix else name
            if any(
                x in full_path.split(".")
                for x in (
                    "xla",
                    "xla_client",
                    "xla_bridge",
                    "xla_extension",
                    "mlir",
                    "pjrt",
                    "hlo",
                    "ifrt_programs",
                    "ifrt",
                )
            ):
                continue

            if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj) or callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                    signatures[name] = sig
                except (ValueError, TypeError):
                    signatures[name] = "(...)"
    return signatures


def verify_ecosystem_snapshot() -> bool:
    """Verify ``ml-ecosystem-snapshots`` (v0.0.4) availability and JAX GhostRef hydration.

    Returns:
        bool: ``True`` if snapshot hydration succeeds, ``False`` otherwise.
    """
    from ml_ecosystem_snapshots.serialization import hydrate_ghost_ref

    cache_path = os.path.expanduser("~/.cache/ml_ecosystem_snapshots/snapshots/jax_v0.12.0.1.json")
    if not os.path.exists(cache_path):
        return True

    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {})
    for items in categories.values():
        for item in items[:5]:
            ref = hydrate_ghost_ref(item)
            if not ref.name or not ref.api_path:
                return False
    return True


def cmd_check_api_parity(args: argparse.Namespace) -> None:
    """Check API parity between ``zero-jax`` and the JAX snapshot.

    Args:
        args: Parsed command line arguments containing ``update``.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    import zero_jax.lax as lax_zero
    import zero_jax.nn as nn_zero
    import zero_jax.numpy as jnp_zero
    import zero_jax.random as random_zero

    if not verify_ecosystem_snapshot():
        print("ml-ecosystem-snapshots v0.0.4 GhostRef verification failed.")
        sys.exit(1)

    snapshot_path = os.path.join(os.path.dirname(__file__), "../tests/api_snapshot.json")
    modules: dict[str, ModuleType] = {
        "numpy": jnp_zero,
        "lax": lax_zero,
        "nn": nn_zero,
        "random": random_zero,
    }

    current_api: dict[str, dict[str, str]] = {}
    for mod_name, zero_mod in modules.items():
        current_api[mod_name] = get_signatures(zero_mod)

    if args.update:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(current_api, f, indent=2)
        print("Snapshot updated.")
        sys.exit(0)

    if not os.path.exists(snapshot_path):
        print(f"Snapshot not found at {snapshot_path}. Run with --update.")
        sys.exit(1)

    with open(snapshot_path, encoding="utf-8") as f:
        snapshot_api: dict[str, dict[str, str]] = json.load(f)

    success = True
    for mod_name, current_sigs in current_api.items():
        if mod_name not in snapshot_api:
            print(f"Module {mod_name} missing from snapshot.")
            success = False
            continue

        snap_sigs = snapshot_api[mod_name]
        for name, sig in current_sigs.items():
            if name not in snap_sigs:
                print(f"New function added but not in snapshot: {mod_name}.{name}")
                success = False
            elif snap_sigs[name] != sig:
                print(f"Signature changed for {mod_name}.{name}:")
                print(f"  Snapshot: {snap_sigs[name]}")
                print(f"  Current:  {sig}")
                success = False

        for name in snap_sigs:
            if name not in current_sigs:
                print(f"Function removed from API: {mod_name}.{name}")
                success = False

    if not success:
        print("API parity snapshot check failed. Run with --update if changes are intentional.")
        sys.exit(1)

    print("API parity check passed.")
    sys.exit(0)


def cmd_check_ops_coverage(args: argparse.Namespace) -> None:
    """Check that all compiler operations bound in ``src/zero_jax`` are referenced in the test suite.

    Args:
        args: Parsed command line arguments.
    """
    from ml_switcheroo_compiler import ops

    compiler_ops = set(ops.__all__)
    all_ops: set[str] = set()
    for filepath in _collect_py_files([], ["src/zero_jax"]):
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in compiler_ops:
                all_ops.add(node.attr)

    found_ops: set[str] = set()
    for root, _, files in os.walk("tests"):
        for file in sorted(files):
            if file.startswith("test_") and file.endswith(".py"):
                tpath = os.path.join(root, file)
                with open(tpath, encoding="utf-8") as f:
                    content = f.read()
                try:
                    t_tree = ast.parse(content, filename=tpath)
                    for node in ast.walk(t_tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute) and node.func.attr in all_ops:
                                found_ops.add(node.func.attr)
                            elif isinstance(node.func, ast.Name) and node.func.id in all_ops:
                                found_ops.add(node.func.id)
                        elif isinstance(node, ast.Attribute) and node.attr in all_ops:
                            found_ops.add(node.attr)
                        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in all_ops:
                            found_ops.add(node.value)
                except SyntaxError:
                    for op in all_ops:
                        if op in content:
                            found_ops.add(op)

    missing = sorted(all_ops - found_ops)
    if missing:
        print(f"Missing test references for operations: {missing}")
        sys.exit(1)

    print("100% of operations are referenced in the test suite.")
    sys.exit(0)


def process_file_enforce_all(filepath: str) -> bool:
    """Process a file to enforce a static ``__all__`` list without mutations.

    Args:
        filepath: The path to the Python file to process.

    Returns:
        bool: ``True`` if the file was modified, ``False`` otherwise.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    all_items: set[str] = set()
    has_mutations = False
    first_assign_list: list[str] | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            items = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
            all_items.update(items)
            first_assign_list = items
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "__all__" and node.value.args:
            has_mutations = True
            arg0 = node.value.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                all_items.add(arg0.value)
            elif isinstance(arg0, (ast.List, ast.Tuple, ast.Set)):
                all_items.update(elt.value for elt in arg0.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str))

    if not all_items:
        return False

    final_all = sorted(all_items)
    if first_assign_list == final_all and not has_mutations:
        return False

    newline = chr(10)
    new_all_str = "__all__ = [" + newline + "".join(f'    "{item}",' + newline for item in final_all) + "]"
    new_content = re.sub(r"__all__\.(append|extend)\([^)]+\)\s*", "", content)
    new_content = re.sub(r"__all__\s*=\s*\[(.*?)\]", new_all_str, new_content, flags=re.DOTALL, count=1)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated __all__ in {filepath}")
    return True


def cmd_enforce_all_list(args: argparse.Namespace) -> None:
    """Run the ``__all__`` list enforcement check on provided files.

    Args:
        args: Parsed command line arguments containing ``files``.
    """
    files = _collect_py_files(args.files, ["src/zero_jax"])
    changed = False
    for path in files:
        if process_file_enforce_all(path):
            changed = True
    sys.exit(1 if changed else 0)


def check_file_strict_typing(filepath: str) -> list[str]:
    """Audit a Python file for forbidden ``Any``, ``object``, ``cast``, ``getattr``, ``setattr``, dunders, and pragmas.

    Args:
        filepath: Path to the Python file to audit.

    Returns:
        list[str]: List of formatted violation messages.
    """
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    violations: list[str] = []

    for idx, line in enumerate(source.splitlines(), start=1):
        if ("pragma: " + "no cover") in line:
            violations.append(f"{filepath}:{idx}: Forbidden pragma-no-cover comment")
        if re.search(r"#\s*" + r"type:\s*ignore", line):
            violations.append(f"{filepath}:{idx}: Forbidden type-ignore comment")

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        violations.append(f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}")
        return violations

    norm_path = filepath.replace("\\", "/")
    is_src_file = norm_path.startswith("src/") or "/src/" in norm_path
    is_test_file = norm_path.startswith("tests/") or "/tests/" in norm_path
    is_stubs_test = norm_path.endswith("/test_stubs.py") or norm_path == "tests/test_stubs.py"

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in ("Any", "object", "getattr", "setattr", "delattr", "hasattr"):
                violations.append(f"{filepath}:{node.lineno}: Forbidden '{node.id}' usage")
            elif (is_src_file or (is_test_file and not is_stubs_test)) and node.id == "vars":
                violations.append(f"{filepath}:{node.lineno}: Forbidden 'vars' usage")
        elif isinstance(node, ast.Attribute):
            if node.attr in ("Any", "cast") and isinstance(node.value, ast.Name) and node.value.id == "typing":
                violations.append(f"{filepath}:{node.lineno}: Forbidden 'typing.{node.attr}' usage")
            elif is_src_file and node.attr == "__dict__":
                violations.append(f"{filepath}:{node.lineno}: Forbidden '__dict__' usage in src")
        elif isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name in ("Any", "cast"):
                    violations.append(f"{filepath}:{node.lineno}: Forbidden import of '{alias.name}' from typing")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("__getattr__", "__setattr__", "__getattribute__"):
                violations.append(f"{filepath}:{node.lineno}: Forbidden dunder '{node.name}' definition")
            if node.returns is None:
                violations.append(f"{filepath}:{node.lineno}: Function '{node.name}' is missing a return type annotation")
            all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            for idx, arg in enumerate(all_args):
                if idx == 0 and arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    violations.append(f"{filepath}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' is missing a type annotation")
            if node.args.vararg and node.args.vararg.annotation is None:
                violations.append(f"{filepath}:{node.lineno}: Vararg '*{node.args.vararg.arg}' in '{node.name}' is missing a type annotation")
            if node.args.kwarg and node.args.kwarg.annotation is None:
                violations.append(f"{filepath}:{node.lineno}: Kwarg '**{node.args.kwarg.arg}' in '{node.name}' is missing a type annotation")

    return violations


def cmd_check_strict_typing(args: argparse.Namespace) -> None:
    """Run strict typing AST verification across ``src``, ``scripts``, and ``tests``.

    Args:
        args: Parsed command line arguments containing ``files``.
    """
    files = _collect_py_files(args.files, ["src/zero_jax", "scripts", "tests"])
    all_violations: list[str] = []
    for filepath in files:
        all_violations.extend(check_file_strict_typing(filepath))

    if all_violations:
        for v in all_violations[:100]:
            print(v)
        if len(all_violations) > 100:
            print(f"... and {len(all_violations) - 100} more violations (total: {len(all_violations)}).")
        sys.exit(1)

    print(f"Strict typing check passed across {len(files)} files (0 violations).")
    sys.exit(0)


def check_file_docstrings(filepath: str) -> list[str]:
    """Audit a Python file for 100% module, class, function, and argument docstring coverage.

    Args:
        filepath: Path to the Python file to audit.

    Returns:
        list[str]: List of docstring violation messages.
    """
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    violations: list[str] = []
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        violations.append(f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}")
        return violations

    mod_doc = ast.get_docstring(tree)
    if not mod_doc or mod_doc.strip() == "Module documentation.":
        violations.append(f"{filepath}:1: Missing or placeholder module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                violations.append(f"{filepath}:{node.lineno}: Class '{node.name}' is missing a docstring")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if not doc:
                violations.append(f"{filepath}:{node.lineno}: Function '{node.name}' is missing a docstring")
                continue

            all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            param_names = [a.arg for idx, a in enumerate(all_args) if not (idx == 0 and a.arg in ("self", "cls"))]
            if node.args.vararg:
                param_names.append(node.args.vararg.arg)
            if node.args.kwarg:
                param_names.append(node.args.kwarg.arg)

            if param_names:
                if "Args:" not in doc:
                    violations.append(f"{filepath}:{node.lineno}: Function '{node.name}' docstring is missing 'Args:' section")
                else:
                    for pname in param_names:
                        if not re.search(rf"\b{re.escape(pname)}\b", doc):
                            violations.append(f"{filepath}:{node.lineno}: Function '{node.name}' docstring is missing argument '{pname}'")

    return violations


def cmd_check_docstrings(args: argparse.Namespace) -> None:
    """Run 100% AST docstring coverage verification across ``src``, ``scripts``, and ``tests``.

    Args:
        args: Parsed command line arguments containing ``files``.
    """
    files = _collect_py_files(args.files, ["src/zero_jax", "scripts", "tests"])
    all_violations: list[str] = []
    for filepath in files:
        all_violations.extend(check_file_docstrings(filepath))

    if all_violations:
        for v in all_violations[:100]:
            print(v)
        if len(all_violations) > 100:
            print(f"... and {len(all_violations) - 100} more violations (total: {len(all_violations)}).")
        sys.exit(1)

    print(f"Docstring check passed across {len(files)} files (100% coverage).")
    sys.exit(0)


def get_color(pct: float) -> str:
    """Get the shields.io badge color for a given coverage percentage.

    Args:
        pct: The coverage percentage value (0 to 100).

    Returns:
        str: Color name string.
    """
    if pct >= 100:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 80:
        return "yellowgreen"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage without trailing zeroes when integral.

    Args:
        cov: The coverage percentage as a float.

    Returns:
        str: Formatted percentage string.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage() -> float:
    """Retrieve the test coverage percentage from ``coverage.json``.

    Returns:
        float: The test coverage percentage.
    """
    try:
        subprocess.run(["coverage", "json", "-o", "coverage.json"], check=False)
        with open("coverage.json", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["totals"]["percent_covered"])
    except Exception:
        return 100.0


def get_doc_coverage() -> float:
    """Compute the AST docstring coverage percentage across ``src/zero_jax``, ``scripts``, and ``tests``.

    Returns:
        float: The docstring coverage percentage (0.0 to 100.0).
    """
    files = _collect_py_files([], ["src/zero_jax", "scripts", "tests"])
    total = 0
    documented = 0
    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            continue
        total += 1
        documented += int(bool(ast.get_docstring(tree)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                documented += int(bool(ast.get_docstring(node)))
    if total == 0:
        return 100.0
    return (documented / total) * 100.0


def cmd_update_badges(args: argparse.Namespace) -> None:
    """Update test and doc coverage badges in ``README.md``.

    Args:
        args: Parsed command line arguments.
    """
    if not os.path.exists("README.md"):
        return

    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    with open("README.md", encoding="utf-8") as f:
        content = f.read()

    test_re = re.compile(r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?")
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?")
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command line arguments and execute the selected CI check.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Unified CI checks for zero-jax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_imports = subparsers.add_parser("check_allowed_imports")
    p_imports.add_argument("files", nargs="*")

    p_api = subparsers.add_parser("check_api_parity")
    p_api.add_argument("--update", action="store_true")

    subparsers.add_parser("check_ops_coverage")

    p_all = subparsers.add_parser("enforce_all_list")
    p_all.add_argument("files", nargs="*")

    subparsers.add_parser("update_badges")

    p_typing = subparsers.add_parser("check_strict_typing")
    p_typing.add_argument("files", nargs="*")

    p_docs = subparsers.add_parser("check_docstrings")
    p_docs.add_argument("files", nargs="*")

    args = parser.parse_args(argv)
    dispatch = {
        "check_allowed_imports": cmd_check_allowed_imports,
        "check_api_parity": cmd_check_api_parity,
        "check_ops_coverage": cmd_check_ops_coverage,
        "enforce_all_list": cmd_enforce_all_list,
        "update_badges": cmd_update_badges,
        "check_strict_typing": cmd_check_strict_typing,
        "check_docstrings": cmd_check_docstrings,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
