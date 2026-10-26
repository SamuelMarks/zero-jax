"""Comprehensive unit and branch coverage tests for the scripts/ package."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import scripts
import scripts.check_allowed_imports as check_allowed_imports
import scripts.check_api_parity as check_api_parity
import scripts.check_ops_coverage as check_ops_coverage
import scripts.generate_todo_plan as generate_todo_plan
import scripts.generate_todo_plan2 as generate_todo_plan2
import scripts.merge_plans as merge_plans
import scripts.run_checks as run_checks
import scripts.update_badges as update_badges

NL = chr(10)


def test_standalone_script_wrappers() -> None:
    """Verify standalone CLI wrapper modules in ``scripts/``."""
    assert scripts.__all__ == []
    assert generate_todo_plan.generate_plan().startswith("# ml-switcheroo-compiler")
    assert generate_todo_plan.count_todo_checkboxes(f"# Header{NL}- [x] Done{NL}- [ ] Todo") == (1, 1)
    assert generate_todo_plan2.generate_plan2().startswith("# ml-switcheroo-compiler")
    assert merge_plans.merge_markdown_plans("# A", "# B") == f"# A{NL}{NL}# B"

    with patch("scripts.check_allowed_imports._run_checks_main") as m1:
        check_allowed_imports.main(["src/zero_jax/__init__.py"])
        m1.assert_called_once_with(["check_allowed_imports", "src/zero_jax/__init__.py"])

    with patch("scripts.check_api_parity._run_checks_main") as m2:
        check_api_parity.main(["--update"])
        m2.assert_called_once_with(["check_api_parity", "--update"])

    with patch("scripts.check_ops_coverage._run_checks_main") as m3:
        check_ops_coverage.check_coverage()
        m3.assert_called_once_with(["check_ops_coverage"])

    with patch("scripts.update_badges._run_checks_main") as m4:
        update_badges.update_readme()
        m4.assert_called_once_with(["update_badges"])


def test_check_file_imports_and_collect_files(tmp_path: Path) -> None:
    """Cover ``check_file_imports``, ``_collect_py_files``, and ``cmd_check_allowed_imports``.

    Args:
        tmp_path: Parameter ``tmp_path``.
    """
    valid_py = tmp_path / "valid.py"
    valid_py.write_text(
        NL.join(
            [
                '"""Doc."""',
                "import os",
                "from . import sibling",
                "from zero_jax import lax",
                "import pydantic",
            ]
        ),
        encoding="utf-8",
    )
    assert run_checks.check_file_imports(str(valid_py)) is True

    bad_syntax = tmp_path / "bad_syntax.py"
    bad_syntax.write_text("def broken(:", encoding="utf-8")
    assert run_checks.check_file_imports(str(bad_syntax)) is False

    disallowed_py = tmp_path / "disallowed.py"
    disallowed_py.write_text(
        NL.join(
            [
                '"""Doc."""',
                "import sys, importlib",
                "import torch",
                "from tensorflow import keras",
                "from . import rel_mod",
                "from os import path",
                'x = sys.modules["torch.nn"]',
                'y = sys.modules["os"]',
                "z = sys.modules[var_name]",
                '__import__("torch")',
                '__import__("os")',
                "__import__(var_name)",
                'importlib.import_module("torch")',
                'sys.modules.get("torch")',
                'builtins.__import__("torch")',
                'a.b.c("x")',
                'a.b.c.d("x")',
            ]
        ),
        encoding="utf-8",
    )
    assert run_checks.check_file_imports(str(disallowed_py)) is False

    sub = tmp_path / "sub"
    sub.mkdir()
    sub_py = sub / "inner.py"
    sub_py.write_text('"""Inner."""', encoding="utf-8")
    collected = run_checks._collect_py_files([str(valid_py), str(sub), str(tmp_path / "nonexistent")], [])
    assert str(valid_py) in collected and str(sub_py) in collected

    with pytest.raises(SystemExit) as exc1:
        run_checks.main(["check_allowed_imports", str(valid_py)])
    assert exc1.value.code == 0

    with pytest.raises(SystemExit) as exc2:
        run_checks.main(["check_allowed_imports", str(disallowed_py)])
    assert exc2.value.code == 1


def test_get_signatures_and_api_parity(tmp_path: Path) -> None:
    """Cover ``get_signatures``, ``verify_ecosystem_snapshot``, and ``cmd_check_api_parity``.

    Args:
        tmp_path: Parameter ``tmp_path``.
    """
    mod = ModuleType("dummy_mod")
    mod.__dict__["valid_fn"] = lambda x: x
    mod.__dict__["_private"] = lambda x: x
    mod.__dict__["xla"] = lambda x: x
    mod.__dict__["not_callable"] = 42

    class _NoSigCallable:
        """Callable whose __signature__ property raises ValueError."""

        @property
        def __signature__(self) -> None:
            """Raise ValueError when signature is inspected."""
            raise ValueError("no sig")

        def __call__(self) -> None:
            """Call dummy."""

    mod.__dict__["no_sig"] = _NoSigCallable()
    sigs = run_checks.get_signatures(mod)
    assert "valid_fn" in sigs and sigs["no_sig"] == "(...)"
    assert run_checks.get_signatures(mod, prefix="xla_client") == {}

    assert run_checks._resolve_zero_jax_endpoint("jax.numpy.sin") is True
    assert run_checks._resolve_zero_jax_endpoint("jax.Array.sum") is True
    assert run_checks._resolve_zero_jax_endpoint("jax.nonexistent_mod.foo") is False
    assert run_checks._resolve_zero_jax_endpoint("jax.numpy.nonexistent_fn") is False

    assert run_checks.verify_ecosystem_snapshot() is True
    with patch("os.path.exists", return_value=False):
        assert run_checks.verify_ecosystem_snapshot() is True

    valid_snap = tmp_path / "valid_snap.json"
    valid_snap.write_text(
        json.dumps({"categories": {"activation": [{"name": "sin", "api_path": "jax.numpy.sin", "kind": "function", "params": []}]}}),
        encoding="utf-8",
    )
    with patch("os.path.expanduser", return_value=str(valid_snap)):
        assert run_checks.verify_ecosystem_snapshot() is True

    fake_snap = tmp_path / "bad_snap.json"
    fake_snap.write_text(
        json.dumps({"categories": {"activation": [{"name": "", "api_path": "", "kind": "function", "params": []}]}}),
        encoding="utf-8",
    )
    with patch("os.path.expanduser", return_value=str(fake_snap)):
        assert run_checks.verify_ecosystem_snapshot() is False

    snap_file = tmp_path / "api_snapshot.json"
    with patch("os.path.join", return_value=str(snap_file)):
        with pytest.raises(SystemExit) as exc_missing:
            run_checks.main(["check_api_parity"])
        assert exc_missing.value.code == 1

        with pytest.raises(SystemExit) as exc_upd:
            run_checks.main(["check_api_parity", "--update"])
        assert exc_upd.value.code == 0

        with pytest.raises(SystemExit) as exc_pass:
            run_checks.main(["check_api_parity"])
        assert exc_pass.value.code == 0

        data = json.loads(snap_file.read_text(encoding="utf-8"))
        del data["random"]
        data["numpy"]["sin"] = "(changed)"
        data["numpy"]["removed_fn"] = "()"
        del data["lax"]["add"]
        snap_file.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_fail:
            run_checks.main(["check_api_parity"])
        assert exc_fail.value.code == 1

    with patch("scripts.run_checks.verify_ecosystem_snapshot", return_value=False):
        with pytest.raises(SystemExit) as exc_vfail:
            run_checks.main(["check_api_parity"])
        assert exc_vfail.value.code == 1


def test_ops_coverage_and_enforce_all_list(tmp_path: Path) -> None:
    """Cover ``cmd_check_ops_coverage``, ``process_file_enforce_all``, and ``cmd_enforce_all_list``.

    Args:
        tmp_path: Parameter ``tmp_path``.
    """
    with pytest.raises(SystemExit) as exc_ops:
        run_checks.main(["check_ops_coverage"])
    assert exc_ops.value.code == 0

    bad_py = tmp_path / "test_bad.py"
    bad_py.write_text("def (add", encoding="utf-8")
    assert run_checks.process_file_enforce_all(str(bad_py)) is False

    with (
        patch("scripts.run_checks._collect_py_files", return_value=[str(bad_py), "src/zero_jax/lax/primitives.py"]),
        patch("os.walk", return_value=[(str(tmp_path), [], ["test_bad.py"])]),
        pytest.raises(SystemExit) as exc_ops_fail,
    ):
        run_checks.main(["check_ops_coverage"])
    assert exc_ops_fail.value.code == 1

    no_all = tmp_path / "no_all.py"
    no_all.write_text(NL.join(['"""Doc."""', "x = 1"]), encoding="utf-8")
    assert run_checks.process_file_enforce_all(str(no_all)) is False

    sorted_all = tmp_path / "sorted_all.py"
    sorted_all.write_text(NL.join(['"""Doc."""', '__all__ = ["a", "b"]']), encoding="utf-8")
    assert run_checks.process_file_enforce_all(str(sorted_all)) is False

    mutated_all = tmp_path / "mutated_all.py"
    mutated_all.write_text(
        NL.join(
            [
                '"""Doc."""',
                '__all__ = ["b", "a"]',
                '__all__.append("c")',
                "__all__.append(123)",
                '__all__.extend(["d", "e"])',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_all1:
        run_checks.main(["enforce_all_list", str(mutated_all)])
    assert exc_all1.value.code == 1

    with pytest.raises(SystemExit) as exc_all2:
        run_checks.main(["enforce_all_list", str(mutated_all)])
    assert exc_all2.value.code == 0


def test_strict_typing_and_docstring_auditors(tmp_path: Path) -> None:
    """Cover ``check_file_strict_typing``, ``cmd_check_strict_typing``, ``check_file_docstrings``, and ``cmd_check_docstrings``.

    Args:
        tmp_path: Parameter ``tmp_path``.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    assert run_checks.check_file_imports("tests/test_scripts.py") is True
    assert run_checks.check_file_imports("scripts/run_checks.py") is True
    assert run_checks.check_file_imports("tests/test_reference_jnp.py") is True
    assert run_checks.check_file_imports("tests/test_stubs.py") is True

    bad_src = src_dir / "bad_mod.py"
    bad_src.write_text(
        NL.join(
            [
                "from typing import " + "Any, cast, List",
                "import typing",
                "x = 1  # pragma: " + "no cover",
                "y = 2  # " + "type: ignore",
                "class UndocClass:",
                "    def __init__(self, a, *args, **kwargs):",
                '        """Doc.',
                "",
                "        Args:",
                "            a: a.",
                "            *args: args.",
                "            **kwargs: kwargs.",
                '        """',
                "        " + "getattr(self, 'a', None)",
                "        " + "setattr(self, 'a', a)",
                "        " + "vars(self)",
                "        _ = self." + "__dict__",
                "        _ = self.other_attr",
                "    def undoc_method(self) -> None:",
                "        return None",
                "def __getattr__(name: " + "object) -> typing." + "Any:",
                '    """Doc without Args section."""',
                "    return typing." + "cast(" + "Any, name)",
                "def bad_args_doc(alpha: int, beta: int, dummy: int = 0) -> int:",
                '    """Doc.',
                "",
                "    Args:",
                "        alpha: Only alpha documented.",
                '    """',
                "    return alpha",
            ]
        ),
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    taut_test = tests_dir / "test_taut.py"
    taut_test.write_text(
        NL.join(
            [
                '"""Doc."""',
                "def test_t() -> None:",
                '    """Doc."""',
                "    a = 1",
                "    check_allclose(a, a)",
            ]
        ),
        encoding="utf-8",
    )
    assert any("Tautological check_allclose" in v for v in run_checks.check_file_strict_typing(str(taut_test)))

    bad_syntax = src_dir / "syntax_err.py"
    bad_syntax.write_text("def (", encoding="utf-8")

    small_bad = src_dir / "small_bad.py"
    small_bad.write_text(
        NL.join(
            [
                '"""Doc."""',
                "class GoodClass:",
                '    """Good class."""',
                "def f_bad(x):",
                "    return x",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_tsmall:
        run_checks.main(["check_strict_typing", str(small_bad)])
    assert exc_tsmall.value.code == 1

    many_violations = src_dir / "many.py"
    many_lines = [f"def f{i}(x):{NL}    return x{NL}" for i in range(110)]
    many_violations.write_text("".join(many_lines), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_t1:
        run_checks.main(["check_strict_typing", str(bad_src), str(bad_syntax), str(many_violations)])
    assert exc_t1.value.code == 1

    with pytest.raises(SystemExit) as exc_dsmall:
        run_checks.main(["check_docstrings", str(small_bad)])
    assert exc_dsmall.value.code == 1

    with pytest.raises(SystemExit) as exc_d1:
        run_checks.main(["check_docstrings", str(bad_src), str(bad_syntax), str(small_bad), str(many_violations)])
    assert exc_d1.value.code == 1

    good_file = tmp_path / "good.py"
    good_file.write_text(
        NL.join(
            [
                '"""Good module."""',
                "",
                "def nop() -> None:",
                '    """No-op."""',
                "    return None",
                "",
                "def add(x: int) -> int:",
                '    """Add.',
                "",
                "    Args:",
                "        x: Value.",
                "",
                "    Returns:",
                "        int: Result.",
                '    """',
                "    return x",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_t2:
        run_checks.main(["check_strict_typing", str(good_file)])
    assert exc_t2.value.code == 0

    with pytest.raises(SystemExit) as exc_d2:
        run_checks.main(["check_docstrings", str(good_file)])
    assert exc_d2.value.code == 0


def test_badges_and_coverage_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover ``get_color``, ``format_cov``, ``get_test_coverage``, ``get_doc_coverage``, and ``cmd_update_badges``.

    Args:
        tmp_path: Parameter ``tmp_path``.
        monkeypatch: Parameter ``monkeypatch``.
    """
    assert run_checks.get_color(100.0) == "brightgreen"
    assert run_checks.get_color(95.0) == "green"
    assert run_checks.get_color(85.0) == "yellowgreen"
    assert run_checks.get_color(75.0) == "yellow"
    assert run_checks.get_color(65.0) == "orange"
    assert run_checks.get_color(50.0) == "red"

    assert run_checks.format_cov(100.0) == "100"
    assert run_checks.format_cov(98.5) == "98.5"

    assert run_checks.get_doc_coverage() == 100.0
    with patch("scripts.run_checks._collect_py_files", return_value=[]):
        assert run_checks.get_doc_coverage() == 100.0

    bad_syntax = tmp_path / "bad_doc.py"
    bad_syntax.write_text("def (", encoding="utf-8")
    with patch("scripts.run_checks._collect_py_files", return_value=[str(bad_syntax)]):
        assert run_checks.get_doc_coverage() == 100.0

    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run"):
        assert run_checks.get_test_coverage() == 100.0
        cov_json = tmp_path / "coverage.json"
        cov_json.write_text(json.dumps({"totals": {"percent_covered": 99.5}}), encoding="utf-8")
        assert run_checks.get_test_coverage() == 99.5

    run_checks.main(["update_badges"])

    readme = tmp_path / "README.md"
    readme.write_text(
        NL.join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-90%25-green.svg)](#)",
            ]
        ),
        encoding="utf-8",
    )
    with patch("scripts.run_checks.get_test_coverage", return_value=100.0), patch("scripts.run_checks.get_doc_coverage", return_value=100.0):
        run_checks.main(["update_badges"])
    updated_readme = readme.read_text(encoding="utf-8")
    assert "test_coverage-100%25-brightgreen.svg" in updated_readme
    assert "doc_coverage-100%25-brightgreen.svg" in updated_readme
