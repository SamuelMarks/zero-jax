import re
import sys
import importlib


def main():
    with open("SEMANTIC_PLAN.md", "r") as f:
        content = f.read()

    modules = {}
    current_module = None

    for line in content.splitlines():
        # Match `jax.module` inside ## `jax.module`
        m_mod = re.match(r"^## `(jax\.[^`]+)`", line)
        if m_mod:
            current_module = m_mod.group(1).replace("jax.", "zero_jax.")
            modules[current_module] = []
            continue

        if current_module:
            m_item = re.match(r"^- \[[xX ]\] ([^:]+):", line)
            if m_item:
                funcs_part = m_item.group(1)
                funcs = [f.strip(" `") for f in funcs_part.split("/")]
                modules[current_module].extend(funcs)

    missing = []
    total = 0
    implemented = 0

    for mod_name, funcs in modules.items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            print(f"Error importing {mod_name}: {e}")
            missing.extend([f"{mod_name}.{f}" for f in funcs])
            total += len(funcs)
            continue

        for f_name in funcs:
            f_name = f_name.strip()
            total += 1
            if hasattr(mod, f_name):
                implemented += 1
            else:
                missing.append(f"{mod_name}.{f_name}")

    pct = (implemented / total) * 100 if total > 0 else 100.0

    # Read current JAX_TODO.md to check if it needs to be updated
    current_todo = ""
    try:
        with open("JAX_TODO.md", "r") as f:
            current_todo = f.read()
    except FileNotFoundError:
        pass

    # Generate the expected JAX_TODO.md content
    new_todo_lines = [
        "# JAX_TODO.md",
        "--- Compliance Report ---",
        f"Overall Compliance: {pct:.1f}%\n",
        "Breakdown by Module:",
    ]
    for mod_name, funcs in modules.items():
        orig_mod_name = mod_name.replace("zero_jax.", "jax.")
        # If any missing, calculate specific module pct
        mod_implemented = sum(
            1 for f_name in funcs if f"{mod_name}.{f_name}" not in missing
        )
        mod_pct = (mod_implemented / len(funcs)) * 100 if funcs else 100.0
        new_todo_lines.append(
            f"  - {orig_mod_name}: {mod_pct:.1f}% ({mod_implemented}/{len(funcs)})"
        )

    if missing:
        new_todo_lines.append("\nMissing APIs:")
        for m in missing:
            new_todo_lines.append(f"  - {m}")

    new_todo = "\n".join(new_todo_lines) + "\n"

    # Always write to update it
    if new_todo != current_todo:
        with open("JAX_TODO.md", "w") as f:
            f.write(new_todo)
        print("Updated JAX_TODO.md.")

    if missing:
        print(
            "COMPLIANCE FAILURE: The following expected APIs are missing or not exported:"
        )
        for m in missing:
            print(f"  - {m}")
        print(f"\nOverall Compliance: {pct:.1f}% ({implemented}/{total})")
        sys.exit(1)
    else:
        print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
        # If the file changed, we want the pre-commit to fail so the user commits the newly generated file
        if new_todo != current_todo:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
