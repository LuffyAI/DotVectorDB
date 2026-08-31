#!/usr/bin/env bash
set -euo pipefail

echo "==> Uninstalling dotvectordb..."
python -m pip uninstall -y dotvectordb

echo "==> Cleaning old builds..."
rm -rf build dist ./*.egg-info

echo "==> Building package..."
python -m build

echo "==> Installing newly built wheel..."
python -m pip install dist/*.whl

echo "==> Done."