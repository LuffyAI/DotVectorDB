If rebuilding:
    python -m pip uninstall -y dotvectordb
    rm -rf build dist *.egg-info

From the root of your PyPI package:
    python -m build

To test the wheel locally:
    python -m pip install dist/*.whl