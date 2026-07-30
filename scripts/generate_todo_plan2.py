import ast
import os
import re

print("# `ml-switcheroo-compiler` JAX Parity Implementation Plan")
print(
    "\nTo ensure `zero-jax` can pass 100% of the official JAX test suite semantically and syntactically, `ml-switcheroo-compiler` must implement the following `ml_switcheroo_compiler.*` operations, mapped back to the JAX APIs that require them.\n"
)

ops_used = set()
cf_used = set()
signal_used = set()
stats_used = set()
random_used = set()
grad_used = set()

for root, _, files in os.walk("src/zero_jax"):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                content = f.read()

                ops_matches = re.findall(r"ops\.([a-zA-Z0-9_]+)", content)
                ops_used.update(ops_matches)

                cf_matches = re.findall(r"cf\.([a-zA-Z0-9_]+)", content)
                cf_used.update(cf_matches)

                signal_matches = re.findall(r"signal\\.([a-zA-Z0-9_]+)", content)
                signal_used.update(signal_matches)

                stats_matches = re.findall(r"stats\\.([a-zA-Z0-9_]+)", content)
                stats_used.update(stats_matches)

                random_matches = re.findall(r"random\.([a-zA-Z0-9_]+)", content)
                random_used.update(random_matches)

                grad_matches = re.findall(r"ir_grad", content)
                if grad_matches:
                    grad_used.add("grad")


def filter_keywords(items):
    keywords = ["xla", "hlo", "pjrt", "mlir"]
    return {item for item in items if not any(kw in item.lower() for kw in keywords)}


ops_used = filter_keywords(ops_used)
cf_used = filter_keywords(cf_used)
signal_used = filter_keywords(signal_used)
stats_used = filter_keywords(stats_used)
random_used = filter_keywords(random_used)
grad_used = filter_keywords(grad_used)


def print_list(title, items, module_prefix):
    if not items:
        return
    print(f"## {title}")
    for item in sorted(list(items)):
        if item in ["creation"]:
            continue
        print(f"- [ ] `{module_prefix}.{item}`")
    print()


print_list("Tensor Operations (`ml_switcheroo_compiler.ops`)", ops_used, "ops")
print_list("Control Flow (`ml_switcheroo_compiler.control_flow`)", cf_used, "cf")
print_list("Signal (`ml_switcheroo_compiler.ops.signal`)", signal_used, "signal")
print_list("Stats (`ml_switcheroo_compiler.ops.stats`)", stats_used, "stats")
print_list(
    "Random Number Generation (`ml_switcheroo_compiler.random`)", random_used, "random"
)
print_list(
    "Automatic Differentiation (`ml_switcheroo_compiler.grad`)", grad_used, "grad"
)

print("## Compiler Infrastructure Requirements\n")
print("- [ ] `LogicalNode`")
print("  - [ ] Must maintain full graph lineage.")
print("- [ ] `ProxyTensor`")
print("  - [ ] Required for `jax.eval_shape` without execution.")
print("- [ ] `evaluate_graph`")
print("  - [ ] Should cache compiled kernels.")
print("- [ ] `Tracing Context`")
print("  - [ ] `_tracer.start_tracing()`")
print("  - [ ] `_tracer.stop_tracing()`")
print("- [ ] `EagerMode`")
print("  - [ ] Required for Python-level control flow.")
