import re

# Read the signature based file
with open("../ml-switcheroo-compiler/docs/planning/JAX_TODO_PLAN.md", "r") as f:
    sig_plan = f.read()

# Read the raw usage file
with open("JAX_TODO_PLAN_TMP2.md", "r") as f:
    raw_plan = f.read()

# Extract all required raw ml_switcheroo_compiler functions
required_ops = set(re.findall(r"\| \[ \] \| `ops\.([a-zA-Z0-9_]+)` \| \|", raw_plan))
required_cf = set(re.findall(r"\| \[ \] \| `cf\.([a-zA-Z0-9_]+)` \| \|", raw_plan))
required_random = set(
    re.findall(r"\| \[ \] \| `random\.([a-zA-Z0-9_]+)` \| \|", raw_plan)
)
required_grad = set(re.findall(r"\| \[ \] \| `grad\.([a-zA-Z0-9_]+)` \| \|", raw_plan))

print("Adding detailed cross-reference section...")

final_content = (
    sig_plan
    + "\n\n## Internal Compiler Bindings Required (`ml_switcheroo_compiler.*`)\n"
)
final_content += "The above JAX primitives map to the following low-level compiler bindings that must be implemented in `ml-switcheroo-compiler`:\n\n"

final_content += "### `ml_switcheroo_compiler.ops`\n"
final_content += "| Status | Binding | Notes |\n|---|---|---|\n"
for op in sorted(list(required_ops)):
    final_content += f"| [ ] | `ops.{op}` | |\n"

final_content += "\n### `ml_switcheroo_compiler.control_flow`\n"
final_content += "| Status | Binding | Notes |\n|---|---|---|\n"
for op in sorted(list(required_cf)):
    final_content += f"| [ ] | `cf.{op}` | |\n"

final_content += "\n### `ml_switcheroo_compiler.random`\n"
final_content += "| Status | Binding | Notes |\n|---|---|---|\n"
for op in sorted(list(required_random)):
    final_content += f"| [ ] | `random.{op}` | |\n"

final_content += "\n### `ml_switcheroo_compiler.grad`\n"
final_content += "| Status | Binding | Notes |\n|---|---|---|\n"
for op in sorted(list(required_grad)):
    final_content += (
        "| [ ] | `ir_grad` | Used as `ml_switcheroo_compiler.grad.grad` |\n"
    )

with open("../ml-switcheroo-compiler/docs/planning/JAX_TODO_PLAN.md", "w") as f:
    f.write(final_content)
