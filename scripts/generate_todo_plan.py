import ast
import os

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
    # Take the first sentence or first line
    first_line = doc.strip().split("\n")[0]
    return first_line.replace("|", "/").strip()


def get_signature(node):
    # Unparse the whole function and get the first line (up to the colon)
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


print("# `ml-switcheroo-compiler` JAX Parity Implementation Plan")
print(
    "\nTo ensure `zero-jax` can pass 100% of the official JAX test suite semantically and syntactically, `ml-switcheroo-compiler` must implement the following operations, matching their expected inputs and behaviors.\n"
)

for category, paths in files_to_scan.items():
    print(f"## {category}\n")
    print("| Status | Name | Signature | Docstring | Notes |")
    print("|---|---|---|---|---|")

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

                print(f"| [ ] | `{node.name}` | `{sig}` | {doc} | |")

print("\n## Compiler Infrastructure Requirements\n")
print("| Status | Name | Signature | Docstring | Notes |")
print("|---|---|---|---|---|")
print(
    "| [ ] | `LogicalNode` | `class LogicalNode(id, op_type, inputs, ...)` | IR/AST Representation for all operations. | Must maintain full graph lineage. |"
)
print(
    "| [ ] | `ProxyTensor` | `class ProxyTensor(id, shape, dtype)` | Abstract tensor proxy for shape/dtype evaluation. | Required for `jax.eval_shape` without execution. |"
)
print(
    "| [ ] | `evaluate_graph` | `evaluate_graph(graph, inputs)` | JIT compilation and graph evaluation pipeline. | Should cache compiled kernels. |"
)
print(
    "| [ ] | `Tracing Context` | `_tracer.start_tracing()`, `_tracer.stop_tracing()` | Context lifecycle management. | Captures dynamic shapes effectively. |"
)
print(
    "| [ ] | `EagerMode` | `with EagerMode(): ...` | Seamless fallback for immediate execution. | Required for Python-level control flow. |"
)
