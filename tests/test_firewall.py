"""The agent is blindfolded. The judge is not.

`robotsim.oracle` is the only module that reads simulator ground truth. If any
module under agent/ or primitives/ can reach it, every number this project
reports is meaningless. This test makes that structural, not a promise.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
BLINDFOLDED = ["agent", "primitives"]
FORBIDDEN_MODULES = {"robotsim.oracle", "harness"}
FORBIDDEN_ATTRS = {"get_base_position", "get_base_orientation", "_sim", "getBasePositionAndOrientation"}


def _python_files(package: str):
    return sorted((REPO / package).rglob("*.py"))


def _imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


@pytest.mark.parametrize("package", BLINDFOLDED)
def test_no_ground_truth_imports(package):
    offenders = []
    for path in _python_files(package):
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _imported_modules(tree):
            if any(module == f or module.startswith(f + ".") for f in FORBIDDEN_MODULES):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert offenders == [], "firewall breach:\n" + "\n".join(offenders)


@pytest.mark.parametrize("package", BLINDFOLDED)
def test_no_ground_truth_attribute_access(package):
    offenders = []
    for path in _python_files(package):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno} touches .{node.attr}")
    assert offenders == [], "firewall breach:\n" + "\n".join(offenders)


def test_blindfolded_packages_are_not_empty():
    for package in BLINDFOLDED:
        assert _python_files(package), f"{package}/ has no modules -- test would pass vacuously"
