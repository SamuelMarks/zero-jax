import re
import os
import ast

# 1. Parse current JAX_TODO_PLAN.md to get statuses
status_map = {}
with open("../ml-switcheroo-compiler/docs/planning/JAX_TODO_PLAN.md", "r") as f:
    for line in f:
        match = re.search(r"\|\s*\[([ xX])\]\s*\|\s*`([^`]+)`", line)
        if match:
            status, name = match.groups()
            status_map[name] = status.lower() == "x"

# 2. Re-run generate_todo_plan logic but with state preservation
files_to_scan = {
    "Tensor Operations (`jax.numpy` / `jax.lax` bindings)": [
        "src/zero_jax/numpy/lax_numpy.py",
        "src/zero_jax/lax/primitives.py",
    ],
    "Control Flow (`jax.lax` control flow)": ["src/zero_jax/lax/control_flow.py"],
    "Random Number Generation (`jax.random`)": ["src/zero_jax/random/prng.py"],
    "Transformations & API (`jax.api`)": ["src/zero_jax/api/transformations.py"],
    "Neural Network Primitives (`jax.nn`)": [
        "src/zero_jax/nn/activation.py",
        "src/zero_jax/nn/initializers.py",
    ],
}


def clean_doc(doc):
    if not doc:
        return "No docstring provided."
    return doc.strip().split("\n")[0].replace("|", "/").strip()


def get_signature(node):
    try:
        source = ast.unparse(node)
        sig_line = source.split("\n")[0]
        if sig_line.startswith("def "):
            sig_line = sig_line[4:]
        if sig_line.endswith(":"):
            sig_line = sig_line[:-1]
        return sig_line
    except Exception:
        return f"{node.name}(...)"


out_lines = []
out_lines.append("# `ml-switcheroo-compiler` JAX Parity Implementation Plan\n")
out_lines.append(
    "To ensure `zero-jax` can pass 100% of the official JAX test suite semantically and syntactically, `ml-switcheroo-compiler` must implement the following operations, matching their expected inputs and behaviors.\n"
)

for category, paths in files_to_scan.items():
    out_lines.append(f"## {category}\n")
    out_lines.append("| Status | Name | Signature | Docstring | Notes |")
    out_lines.append("|---|---|---|---|---|")
    seen = set()
    for filepath in paths:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if node.name in seen:
                    continue
                seen.add(node.name)
                sig = get_signature(node)
                doc = clean_doc(ast.get_docstring(node))
                status_char = "x" if status_map.get(node.name, False) else " "
                out_lines.append(
                    f"| [{status_char}] | `{node.name}` | `{sig}` | {doc} | |"
                )
    out_lines.append("")

out_lines.append("## Compiler Infrastructure Requirements\n")
out_lines.append("| Status | Name | Signature | Docstring | Notes |")
out_lines.append("|---|---|---|---|---|")
for req in [
    (
        "LogicalNode",
        "`class LogicalNode(id, op_type, inputs, ...)`",
        "IR/AST Representation for all operations.",
        "Must maintain full graph lineage.",
    ),
    (
        "ProxyTensor",
        "`class ProxyTensor(id, shape, dtype)`",
        "Abstract tensor proxy for shape/dtype evaluation.",
        "Required for `jax.eval_shape` without execution.",
    ),
    (
        "evaluate_graph",
        "`evaluate_graph(graph, inputs)`",
        "JIT compilation and graph evaluation pipeline.",
        "Should cache compiled kernels.",
    ),
    (
        "Tracing Context",
        "`_tracer.start_tracing()`, `_tracer.stop_tracing()`",
        "Context lifecycle management.",
        "Captures dynamic shapes effectively.",
    ),
    (
        "EagerMode",
        "`with EagerMode(): ...`",
        "Seamless fallback for immediate execution.",
        "Required for Python-level control flow.",
    ),
]:
    status_char = "x" if status_map.get(req[0], False) else " "
    out_lines.append(
        f"| [{status_char}] | `{req[0]}` | {req[1]} | {req[2]} | {req[3]} |"
    )
out_lines.append("")

# 3. Read required bindings from JAX_TODO_PLAN_TMP2.md
required_ops = set()
required_cf = set()
required_random = set()
required_grad = set()
with open("JAX_TODO_PLAN_TMP2.md", "r") as f:
    raw_plan = f.read()
required_ops.update(
    re.findall(r"\|\s*\[.*?\]\s*\|\s*`ops\.([a-zA-Z0-9_]+)`\s*\|", raw_plan)
)
required_cf.update(
    re.findall(r"\|\s*\[.*?\]\s*\|\s*`cf\.([a-zA-Z0-9_]+)`\s*\|", raw_plan)
)
required_random.update(
    re.findall(r"\|\s*\[.*?\]\s*\|\s*`random\.([a-zA-Z0-9_]+)`\s*\|", raw_plan)
)
required_grad.update(
    re.findall(r"\|\s*\[.*?\]\s*\|\s*`grad\.([a-zA-Z0-9_]+)`\s*\|", raw_plan)
)

out_lines.append("## Internal Compiler Bindings Required (`ml_switcheroo_compiler.*`)")
out_lines.append(
    "The above JAX primitives map to the following low-level compiler bindings that must be implemented in `ml-switcheroo-compiler`:\n"
)

out_lines.append("### `ml_switcheroo_compiler.ops`")
out_lines.append("| Status | Binding | Notes |")
out_lines.append("|---|---|---|")
for op in sorted(list(required_ops)):
    status_char = "x" if status_map.get(f"ops.{op}", False) else " "
    out_lines.append(f"| [{status_char}] | `ops.{op}` | |")
out_lines.append("")

out_lines.append("### `ml_switcheroo_compiler.control_flow`")
out_lines.append("| Status | Binding | Notes |")
out_lines.append("|---|---|---|")
for op in sorted(list(required_cf)):
    status_char = "x" if status_map.get(f"cf.{op}", False) else " "
    out_lines.append(f"| [{status_char}] | `cf.{op}` | |")
out_lines.append("")

out_lines.append("### `ml_switcheroo_compiler.random`")
out_lines.append("| Status | Binding | Notes |")
out_lines.append("|---|---|---|")
for op in sorted(list(required_random)):
    status_char = "x" if status_map.get(f"random.{op}", False) else " "
    out_lines.append(f"| [{status_char}] | `random.{op}` | |")
out_lines.append("")

out_lines.append("### `ml_switcheroo_compiler.grad`")
out_lines.append("| Status | Binding | Notes |")
out_lines.append("|---|---|---|")
for op in sorted(list(required_grad)):
    status_char = "x" if status_map.get("ir_grad", False) else " "
    out_lines.append(
        f"| [{status_char}] | `ir_grad` | Used as `ml_switcheroo_compiler.grad.grad` |"
    )
out_lines.append("")

with open("../ml-switcheroo-compiler/docs/planning/JAX_TODO_PLAN.md", "w") as f:
    f.write("\n".join(out_lines))
