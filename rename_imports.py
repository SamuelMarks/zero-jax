import os
import re

for root, _, files in os.walk("src/zero_jax"):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                content = f.read()

            # Replace exactly `import ml_switcheroo_compiler`
            content = re.sub(
                r"^(\s*)import\s+ml_switcheroo_compiler(\s*)$",
                r"\1import ml_switcheroo_compiler as ml_switcheroo_compiler\2",
                content,
                flags=re.MULTILINE,
            )

            # Replace `import ml_switcheroo_compiler.` -> `import ml_switcheroo_compiler.`
            content = re.sub(
                r"import\s+ml_switcheroo_compiler\.",
                r"import ml_switcheroo_compiler.",
                content,
            )

            # Replace `from ml_switcheroo_compiler ` -> `from ml_switcheroo_compiler `
            content = re.sub(
                r"from\s+ml_switcheroo_compiler\s+",
                r"from ml_switcheroo_compiler ",
                content,
            )

            # Replace `from ml_switcheroo_compiler.` -> `from ml_switcheroo_compiler.`
            content = re.sub(
                r"from\s+ml_switcheroo_compiler\.",
                r"from ml_switcheroo_compiler.",
                content,
            )

            # Additional replacement for `ml_switcheroo_compiler.Tensor` just in case
            content = re.sub(
                r"ml_switcheroo_compiler\.Tensor",
                r"ml_switcheroo_compiler.Tensor",
                content,
            )
            # Wait, if we did `import ml_switcheroo_compiler as ml_switcheroo_compiler`, `ml_switcheroo_compiler.Tensor` will still work!
            # I will only replace `ml_switcheroo_ir` -> `ml_switcheroo_ir`? No, the package name is `ml_switcheroo_compiler`. Wait, `ml_switcheroo_ir` is just `ml_switcheroo_compiler.ir` or what?

            with open(path, "w") as f:
                f.write(content)
