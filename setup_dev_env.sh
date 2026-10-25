#!/bin/bash
set -e
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
echo "Setting up local editable environment..."
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-switcheroo-compiler
pip install "ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl" || pip install -e ../ml-ecosystem-snapshots
pip install --no-deps "griffelib @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffelib" "griffecli @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffecli" || true
pip install -e ".[test]"
echo "Done!"
