#!/bin/bash
set -e
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
echo "Setting up local editable environment..."
pip install "ml-switcheroo-ir @ https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.4/ml_switcheroo_ir-0.0.4-py3-none-any.whl"
pip install "ml-switcheroo-compiler @ https://github.com/SamuelMarks/ml-switcheroo-compiler/releases/download/v0.0.1/ml_switcheroo_compiler-0.0.1-py3-none-any.whl"
pip install "ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl"
pip install --no-deps "griffelib @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffelib" "griffecli @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffecli" || true
for pkg in ../zero-chex ../zero-flax ../zero-grain ../zero-optax ../zero-orbax; do
  if [ -d "$pkg" ]; then
    pip install --no-deps -e "$pkg"
  fi
done
pip install -e ".[test]"
echo "Done!"
