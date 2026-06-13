import ast
import os
import re

print("# `ml-switcheroo-compiler` JAX Parity Implementation Plan")
print(
    "\nTo ensure `zero-jax` can pass 100% of the official JAX test suite semantically and syntactically, `ml-switcheroo-compiler` must implement the following `ml_switcheroo_compiler.*` operations, mapped back to the JAX APIs that require them.\n"
)

# This script directly extracts what `ml_switcheroo_compiler` modules are being called.
ops_used = set()
cf_used = set()
random_used = set()
grad_used = set()

# Scan all python files in src/zero_jax
for root, _, files in os.walk("src/zero_jax"):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                content = f.read()

                ops_matches = re.findall(r"ops\.([a-zA-Z0-9_]+)", content)
                ops_used.update(ops_matches)

                cf_matches = re.findall(r"cf\.([a-zA-Z0-9_]+)", content)
                cf_used.update(cf_matches)

                random_matches = re.findall(r"random\.([a-zA-Z0-9_]+)", content)
                random_used.update(random_matches)

                grad_matches = re.findall(r"ir_grad", content)
                if grad_matches:
                    grad_used.add("grad")


def print_table(title, items, module_prefix):
    if not items:
        return
    print(f"## {title}")
    print()
    print("| Status | Required Implementation | Notes |")
    print("|---|---|---|")
    for item in sorted(list(items)):
        if item in ["creation"]:
            continue
        print(f"| [ ] | `{module_prefix}.{item}` | |")
    print()


print_table("Tensor Operations (`ml_switcheroo_compiler.ops`)", ops_used, "ops")
print_table("Control Flow (`ml_switcheroo_compiler.control_flow`)", cf_used, "cf")
print_table(
    "Random Number Generation (`ml_switcheroo_compiler.random`)", random_used, "random"
)
print_table(
    "Automatic Differentiation (`ml_switcheroo_compiler.grad`)", grad_used, "grad"
)


print("\n## Compiler Infrastructure Requirements\n")
print("| Status | Required Implementation | Notes |")
print("|---|---|---|")
print("| [ ] | `LogicalNode` | Must maintain full graph lineage. |")
print("| [ ] | `ProxyTensor` | Required for `jax.eval_shape` without execution. |")
print("| [ ] | `evaluate_graph` | Should cache compiled kernels. |")
print(
    "| [ ] | `Tracing Context` | `_tracer.start_tracing()`, `_tracer.stop_tracing()` |"
)
print("| [ ] | `EagerMode` | Required for Python-level control flow. |")
