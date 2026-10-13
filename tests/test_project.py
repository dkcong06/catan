"""Check the Stage 1 package boundaries in a fresh interpreter."""

import sys
from subprocess import run


def test_all_implemented_packages_import_without_machine_learning() -> None:
    script = """
import importlib
import pkgutil
import sys
import catan
import rl
for package in (catan, rl):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(module.name)
assert 'torch' not in sys.modules
assert 'torch_geometric' not in sys.modules
assert 'dgl' not in sys.modules
"""
    result = run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
