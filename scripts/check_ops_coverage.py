import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler.ops as ops


def check_coverage():
    all_ops = set(ops.__all__)
    ignored = {
        "OpDef",
        "AssignVariable",
        "ReadVariable",
        "get_op",
        "register_op",
        "pi",
        "ndarray",
        "array",
        "asarray",
        "broadcast_shapes",
        "expand_dims",
        "logspace",
    }
    all_ops = all_ops - ignored

    # Simple AST or regex check for now, ensuring all ops are named somewhere in tests
    import re

    found_ops = set()
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    for op in all_ops:
                        if re.search(r"\b" + op + r"\b", content):
                            found_ops.add(op)

    missing = all_ops - found_ops
    if missing:
        print(f"Missing ops in tests: {len(missing)}")
        print(sorted(list(missing)))
        sys.exit(1)

    print("100% of operations are referenced in the test suite.")
    sys.exit(0)


if __name__ == "__main__":
    check_coverage()
