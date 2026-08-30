"""The agent is blindfolded. The judge is not.

`robotsim.oracle` is the only module that reads simulator ground truth, and
`robotsim.world` holds the scene layout that produced it. If any module under
agent/ or primitives/ can reach either, every number this project reports is
meaningless. This test makes that structural, not a promise.

This static scan and the `RobotIO` surface test (Task 5) are two halves of one
guarantee, not redundant copies of it: runtime dependency injection and
re-export through a permitted module are both invisible to an AST scan, so
deleting either half leaves a hole the other never covered.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
BLINDFOLDED = ["agent", "primitives"]

FORBIDDEN_MODULES = {"robotsim.oracle", "robotsim.world", "harness"}

# `sim` and `physics_client` are the raw PyBullet handles; everything reachable
# through them is ground truth. The prefixes cover the pose/proprioception
# getters wholesale -- agent-side code reads joints via `io.joint_positions()`.
# `.oracle` and `.world` are NOT redundant with FORBIDDEN_MODULES. Python binds a
# submodule as an attribute of its parent package as soon as ANY module in the
# process imports it, and harness/ imports robotsim.oracle on every run. So a file
# holding only a clean `import robotsim` can still reach ground truth via
# `robotsim.oracle.get_object_pose(...)` -- and the module scan cannot see it,
# because the import that creates the binding lives in harness/, not in the file
# being scanned. Only the attribute scan can catch this.
FORBIDDEN_ATTRS = {"sim", "physics_client", "_bowl_centers", "oracle", "world",
                   "getBasePositionAndOrientation", "getLinkState", "getContactPoints"}
FORBIDDEN_ATTR_PREFIXES = ("get_base_", "get_link_", "get_joint_", "physics_client")

# Deliberately shallow. This firewall defends against accidental architectural
# drift by a cooperating author, not against a saboteur: a determined author can
# defeat any AST scan. Constant folding, taint analysis, and tracking aliases of
# these builtins are NOT attempted, on purpose -- a half-built version of that
# would buy confidence it cannot back up.
DYNAMIC_CALLS = {"__import__", "eval", "exec"}
DYNAMIC_MODULES = {"importlib"}


def _python_files(package: str):
    return sorted((REPO / package).rglob("*.py"))


def _imported_names(tree: ast.AST):
    """Yield (lineno, dotted_name) for every symbol path an import can bind."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.lineno, node.module
                for alias in node.names:
                    yield node.lineno, f"{node.module}.{alias.name}"
            else:
                # `from . import harness` / `from .. import harness`
                for alias in node.names:
                    yield node.lineno, alias.name


def _is_forbidden_module(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN_MODULES)


def _is_forbidden_attr(attr: str) -> bool:
    return attr in FORBIDDEN_ATTRS or attr.startswith(FORBIDDEN_ATTR_PREFIXES)


def _module_breaches(tree: ast.AST):
    return [(lineno, f"imports {name}")
            for lineno, name in _imported_names(tree) if _is_forbidden_module(name)]


def _attr_breaches(tree: ast.AST):
    return [(node.lineno, f"touches .{node.attr}") for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and _is_forbidden_attr(node.attr)]


def _dynamic_breaches(tree: ast.AST):
    hits = [(lineno, f"imports {name}") for lineno, name in _imported_names(tree)
            if name.split(".")[0] in DYNAMIC_MODULES]
    hits += [(node.lineno, f"uses {node.id}") for node in ast.walk(tree)
             if isinstance(node, ast.Name) and node.id in DYNAMIC_CALLS]
    return hits


def _scan(package: str, detector):
    offenders = []
    for path in _python_files(package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, what in detector(tree):
            offenders.append(f"{path.relative_to(REPO)}:{lineno} {what}")
    return offenders


@pytest.mark.parametrize("package", BLINDFOLDED)
def test_no_ground_truth_imports(package):
    offenders = _scan(package, _module_breaches)
    assert offenders == [], "firewall breach:\n" + "\n".join(offenders)


@pytest.mark.parametrize("package", BLINDFOLDED)
def test_no_ground_truth_attribute_access(package):
    offenders = _scan(package, _attr_breaches)
    assert offenders == [], "firewall breach:\n" + "\n".join(offenders)


@pytest.mark.parametrize("package", BLINDFOLDED)
def test_no_dynamic_imports(package):
    offenders = _scan(package, _dynamic_breaches)
    assert offenders == [], "dynamic import escape hatch:\n" + "\n".join(offenders)


def test_robotsim_init_does_not_reexport():
    """`import robotsim` is legal agent-side, so the package __init__ must not
    pull the oracle or the world in behind the firewall's back."""
    tree = ast.parse((REPO / "robotsim" / "__init__.py").read_text(encoding="utf-8"))
    assert not list(_imported_names(tree)), "robotsim/__init__.py must stay import-free"


def test_blindfolded_packages_are_not_empty():
    """Catches a package renamed or moved out from under the scan -- if agent/
    disappeared, every scan above would go green by scanning nothing. It does
    NOT prove the scan works on real code; the positive controls below do that.
    """
    for package in BLINDFOLDED:
        assert _python_files(package), f"{package}/ is missing or has no modules"


# --- positive controls -------------------------------------------------------
# Every assertion above is `offenders == []`, which a gutted detector satisfies
# forever. These run the SAME predicates against sources that must and must not
# trip, so a detector that stops detecting fails here loudly.

BREACHING_IMPORTS = [
    "import harness",
    "from harness import run",
    "from harness.report import main",
    "from robotsim.oracle import get_pose",
    "from robotsim.world import World",
    "import robotsim.oracle as truth",
    "from .. import harness",
    "def peek():\n    from robotsim import oracle\n    return oracle",
]

CLEAN_IMPORTS = [
    "import numpy as np",
    "import robotsim",
    "from robotsim import io",
    "from robotsim.io import RobotIO",
    "from primitives.grasp import grasp",
    "import harness_utils",          # near-miss: prefix match must need the dot
    "from . import prompts",
]

BREACHING_ATTRS = [
    "handle = env.sim",
    "handle = env.physics_client",
    "centers = scene._bowl_centers",
    "xyz = block.get_base_position()",
    "quat = block.get_base_orientation()",
    "state = body.get_link_state(3)",
    "q = robot.get_joint_angle(0)",
    "p.getBasePositionAndOrientation(0)",
    "p.getLinkState(body, 3)",
    "p.getContactPoints(a, b)",
    # the package-attribute tunnel: `import robotsim` alone reads as clean
    "pose = robotsim.oracle.get_object_pose('b')",
    "bounds = robotsim.world.WORKSPACE",
]

CLEAN_ATTRS = [
    "q = io.joint_positions()",
    "lo, hi = io.workspace_bounds()",
    "w = io.gripper_width()",
    "img = io.camera_rgb()",
    "d = np.linalg.norm(v)",
]

BREACHING_DYNAMIC = [
    "mod = __import__('harness')",
    "eval('robotsim.oracle')",
    "exec('import harness')",
    "import importlib",
    "from importlib import import_module",
]

CLEAN_DYNAMIC = [
    "n = int('3')",
    "import json",
    "data = json.loads(s)",
]


@pytest.mark.parametrize("source", BREACHING_IMPORTS)
def test_import_detector_catches_breach(source):
    assert _module_breaches(ast.parse(source)), f"detector blind to: {source!r}"


@pytest.mark.parametrize("source", CLEAN_IMPORTS)
def test_import_detector_allows_clean(source):
    assert _module_breaches(ast.parse(source)) == [], f"false positive on: {source!r}"


@pytest.mark.parametrize("source", BREACHING_ATTRS)
def test_attr_detector_catches_breach(source):
    assert _attr_breaches(ast.parse(source)), f"detector blind to: {source!r}"


@pytest.mark.parametrize("source", CLEAN_ATTRS)
def test_attr_detector_allows_clean(source):
    assert _attr_breaches(ast.parse(source)) == [], f"false positive on: {source!r}"


@pytest.mark.parametrize("source", BREACHING_DYNAMIC)
def test_dynamic_detector_catches_breach(source):
    assert _dynamic_breaches(ast.parse(source)), f"detector blind to: {source!r}"


@pytest.mark.parametrize("source", CLEAN_DYNAMIC)
def test_dynamic_detector_allows_clean(source):
    assert _dynamic_breaches(ast.parse(source)) == [], f"false positive on: {source!r}"
