# Self-Verifying Robot Arm Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VLM-driven simulated robot arm that verifies its own work after every step — via error strings, gripper proprioception, and a fresh camera image — recovers from failures, escalates ambiguity to a human, and prove the improvement over a blind open-loop baseline with a one-command reproducible eval harness.

**Architecture:** A PyBullet/panda-gym world builds scenes declaratively from `scenes.yaml`. The agent's entire sensory world is a `RobotIO` facade (camera pixels, EE pose, joint states, finger aperture) plus error strings from five primitives. Object positions are recovered by classical HSV segmentation on rendered pixels and unprojected to world coordinates by *our* code — the VLM only ever names object IDs. A separate `robotsim/oracle.py` reads sim ground truth and is imported **only** by `harness/`; an automated AST test fails the build if `agent/` or `primitives/` ever touch it. Baseline and agent share every line of that machinery and differ only in the control loop.

**Tech Stack:** Python 3.11.6 · pybullet 3.2.6 · panda-gym 3.0.7 · gymnasium 0.29.1 · numpy 1.26.4 · opencv-python-headless 4.10 · google-genai 2.20.0 · model `gemini-robotics-er-2-preview` (Gemini Robotics-ER 2, embodied-reasoning VLM) · pytest 8.3.3

---

## Decisions already locked (do not relitigate)

| Decision | Value | Why |
|---|---|---|
| Simulator | panda-gym 3.0.7 on pybullet 3.2.6, `renderer="Tiny"`, DIRECT mode | Spike proved headless CPU, scripted grasp, RGB render in 4.4 s wall time |
| macOS build fix | `CFLAGS="-std=gnu17 -Dfdopen=fdopen"` when building pybullet | Bundled zlib does `#define fdopen(fd,mode) NULL`, collides with the macOS SDK `fdopen` declaration; without this the wheel does not build on Apple Silicon |
| VLM | `gemini-robotics-er-2-preview` via `client.interactions.create` | Embodied-reasoning VLM with vision + function calling + thinking; free tier lets judges run live at zero cost; $2/$10 per MTok paid |
| Determinism | `generation_config={"seed": 0, "thinking_level": "low"}` | The Interactions API exposes **`seed`, not `temperature`**. README must say so — do not claim "temperature 0" |
| Eval scale | 10 scenes × 3 seeds = 30 episodes per condition | User decision |
| Physics | 5 cm block, `lateral_friction=2.0`, mass 0.08, slow servo; target ≥95% success for a *correct* grasp pose | User decision. Realism is a stated non-goal |
| Replay cache key | `(scene_id, condition, seed, step_index, call_kind)` — **not** image bytes | Image bytes are not bit-identical across machines; keying on them would break `make judge` on a judge's laptop. Prompt hash is stored alongside and reported as a "replay drift" count |

---

## File Structure

```
m1-assignment/
├── Makefile                       setup / spike / baseline / agent / judge / judge-live / test / report
├── README.md                      user + bottleneck, before/after split, firewall, citations, hot take
├── REPRODUCTION.md                clean-machine guide: commands, versions, runtime, cost
├── IMPROVEMENT_CHANGELOG.md       one row per experiment, written as we go
├── SECRETS.example / SECRETS      GEMINI_API_KEY (SECRETS is gitignored)
├── requirements.txt               pinned
├── scenes.yaml                    all 10 scenes, declarative
├── spike/spike.py                 DONE — feasibility proof
│
├── robotsim/                      the world. Only harness/ may import oracle.
│   ├── cameras.py                 Camera dataclass, view/proj matrices, render(), unproject()
│   ├── world.py                   builds a scene from a SceneSpec; owns the pybullet handle
│   ├── io.py                      RobotIO facade — the ONLY thing the agent side may hold
│   └── oracle.py                  GROUND TRUTH. success predicates. harness-only.
│
├── primitives/                    the robot's hands. agent-safe.
│   ├── perception.py              HSV segmentation on pixels -> Detection list
│   ├── feedback.py                Feedback dataclass
│   └── api.py                     look / move_to / grasp / place / ask_human / report_done
│
├── agent/                         agent-safe.
│   ├── llm.py                     Gemini client + replay cache + token/cost accounting
│   ├── prompts.py                 shared system preamble, planner prompt, verifier prompt
│   ├── memory.py                  EpisodeMemory — in-context attempt log
│   ├── verify.py                  L1 / L2 / L3 as independent toggles
│   ├── baseline.py                one-shot plan, execute blind, claim success
│   └── react.py                   ReAct loop with budgets
│
├── harness/                       the proof machine. The only importer of oracle.
│   ├── scenes.py                  scenes.yaml loader -> SceneSpec
│   ├── run.py                     {conditions} × {scenes} × {seeds}, headless
│   ├── metrics.py                 EpisodeResult, aggregation, honesty gap
│   ├── report.py                  results/report.md + results/report.html + inline SVG chart
│   └── trajectory.py              results/trajectories/<episode>.html
│
├── tests/
│   ├── test_firewall.py           AST scan: agent/ + primitives/ never import oracle
│   ├── test_cameras.py            project/unproject round-trip
│   ├── test_perception.py         segmentation on a rendered frame
│   ├── test_primitives.py         grasp reliability ≥95%, error strings
│   └── test_replay.py             cache hit/miss semantics
│
├── cache/                         committed replay cache — makes `make judge` free
└── results/                       generated. never hand-edited.
```

---

## Task 1: Package skeleton + Makefile + firewall test

**Files:**
- Create: `robotsim/__init__.py`, `primitives/__init__.py`, `agent/__init__.py`, `harness/__init__.py`, `tests/__init__.py`
- Create: `Makefile`
- Test: `tests/test_firewall.py`

- [ ] **Step 1: Write the failing firewall test**

```python
# tests/test_firewall.py
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
# AMENDED after Task 1 quality review -- robotsim.world was the shortest path to
# ground truth (World exposes .sim and ._bowl_centers) and was not blocked.
# Safe to block: robotsim/io.py is the only module that legitimately needs World,
# and it lives inside robotsim/, which is never scanned.
FORBIDDEN_MODULES = {"robotsim.oracle", "robotsim.world", "harness"}
FORBIDDEN_ATTRS = {"sim", "physics_client", "_bowl_centers",
                   "getBasePositionAndOrientation", "getLinkState", "getContactPoints",
                   # `oracle` and `world` are here, not only in FORBIDDEN_MODULES,
                   # because Python binds a submodule onto its parent package as soon
                   # as ANYTHING in the process imports it -- and harness/ imports
                   # robotsim.oracle on every run. So a plain `import robotsim` in
                   # agent/ followed by `robotsim.oracle.get_object_pose(...)` reaches
                   # real ground truth, and the module scan cannot see it: the import
                   # that creates the binding lives in a different file.
                   "oracle", "world"}
# Prefix matching, so a ground-truth accessor added upstream by panda-gym is blocked
# by default rather than by remembering to update this file.
FORBIDDEN_ATTR_PREFIXES = ("get_base_", "get_link_", "get_joint_")


def _python_files(package: str):
    return sorted((REPO / package).rglob("*.py"))


def _imported_names(tree: ast.AST):
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
        for module in _imported_names(tree):
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/pytest tests/test_firewall.py -v`
Expected: FAIL — `test_blindfolded_packages_are_not_empty` fails because `agent/` and `primitives/` are still empty.

- [ ] **Step 3: Create the package markers**

```bash
for p in robotsim primitives agent harness tests; do
  printf '' > "$p/__init__.py"
done
```

- [ ] **Step 4: Write the Makefile**

```makefile
PY := .venv/bin/python
PYTEST := .venv/bin/pytest
UV := uv

# pybullet's bundled zlib does `#define fdopen(fd,mode) NULL`, which collides with
# the macOS SDK declaration of fdopen. Overriding the macro is what makes it build.
BUILD_ENV := CFLAGS="-std=gnu17 -Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"

.PHONY: setup spike test baseline agent judge judge-live report clean

setup:
	$(UV) venv --python 3.11 .venv
	$(BUILD_ENV) VIRTUAL_ENV=.venv $(UV) pip install -r requirements.txt
	@echo "setup done. copy SECRETS.example -> SECRETS and add GEMINI_API_KEY for live runs."

spike:
	$(PY) spike/spike.py

test:
	$(PYTEST) tests/ -v

baseline:
	$(PY) -m harness.run --conditions baseline --mode replay

agent:
	$(PY) -m harness.run --conditions agent --mode replay

# The headline target. Runs the ENTIRE eval offline, free, from the committed cache.
judge:
	$(PY) -m harness.run --conditions all --mode replay
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

judge-live:
	$(PY) -m harness.run --conditions all --mode live
	$(PY) -m harness.report

report:
	$(PY) -m harness.report

clean:
	rm -rf results/* spike/out/*
```

- [ ] **Step 5: Run the test again**

Run: `.venv/bin/pytest tests/test_firewall.py -v`
Expected: PASS. After the quality-review hardening below this is **46 tests** — the
real scans plus the positive-control corpora. The count is expected to grow, not stay
fixed; what matters is that the control tests prove the detector still detects.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: project skeleton, Makefile, ground-truth firewall test"
```

> **Amendments from the Task 1 quality review** (all implemented; listed here so later
> spec reviews compare against the real contract):
> 1. `robotsim.world` added to `FORBIDDEN_MODULES` — CRITICAL, it was the shortest
>    path to ground truth from agent-side code.
> 2. `robotsim/__init__.py` must stay import-free, guarded by its own test —
>    `import robotsim` is legal agent-side, so a re-export there would bypass the scan.
> 3. **Positive-control tests.** Every original assertion was `assert offenders == []`,
>    so a gutted detector would have passed forever and the project's headline claim
>    would have quietly become false. Parametrized `BREACHES` / `CLEAN` corpora now
>    prove the detector detects, sharing one predicate with the real scan.
> 4. `FORBIDDEN_ATTRS` rewritten with prefix matching; `_sim` was dead (`World`
>    exposes `.sim`, public).
> 5. `encoding="utf-8"` on every `read_text()` — these sources contain em dashes and
>    the test would crash under `LC_ALL=C` rather than report a breach.
> 6. `from .. import harness` now caught; import breaches now report line numbers.
> 7. A deliberately-simple dynamic-import guard (`__import__`/`eval`/`exec`/`importlib`),
>    with a comment stating that this firewall defends against accidental architectural
>    drift by a cooperating author, not against a saboteur.
> 8. Makefile: `judge: test` (the headline target proves the firewall before printing
>    numbers), `.DEFAULT_GOAL := help` plus a `help` target so a bare `make` cannot
>    rebuild the venv, and the report-path echo copied to `judge-live`.
>
> **Two gaps an AST scan structurally cannot close**, both covered by Task 5 instead:
> runtime dependency injection (passing a `World` where a `RobotIO` is expected) and
> re-export through a permitted module. `test_surface_is_exactly_the_allowed_set` and
> `test_io_cannot_reach_object_poses` are the other half of this guarantee — neither
> half may be deleted on the assumption that the other covers it.

---

## Task 2: Cameras — matrices we own, and pixel→world unprojection

**Files:**
- Create: `robotsim/cameras.py`
- Test: `tests/test_cameras.py`

Rationale: panda-gym's `sim.render()` builds its own view/projection matrices internally and hands back only pixels. We need the *same* matrices to turn a detected pixel into a world coordinate. So we compute them ourselves and pass them to pybullet directly. This is the load-bearing piece of "the VLM names objects, our code computes poses".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cameras.py
import numpy as np
import pytest

from robotsim.cameras import OVERHEAD, OBLIQUE, Camera


@pytest.mark.parametrize("cam", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
@pytest.mark.parametrize("point", [
    (0.00, 0.00, 0.025),
    (0.10, -0.12, 0.025),
    (-0.08, 0.15, 0.025),
])
def test_project_unproject_roundtrip(cam: Camera, point):
    world = np.array(point)
    px, py = cam.project(world)
    assert 0 <= px < cam.width and 0 <= py < cam.height, "test point fell outside the frame"
    recovered = cam.unproject(px, py, z_plane=world[2])
    assert np.allclose(recovered, world, atol=3e-3), f"{recovered} != {world}"


def test_unproject_is_pure_geometry():
    """unproject must not need the simulator -- it is camera math, not a ground-truth lookup."""
    recovered = OVERHEAD.unproject(OVERHEAD.width // 2, OVERHEAD.height // 2, z_plane=0.0)
    assert recovered.shape == (3,)
    assert abs(recovered[2] - 0.0) < 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_cameras.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'robotsim.cameras'`

- [ ] **Step 3: Implement `robotsim/cameras.py`**

```python
"""Camera model. Owns its view/projection matrices so that pixel -> world is our math.

Two views, deliberately:
  OVERHEAD  near-nadir, used for detection + unprojection (a nadir ray hits the
            table plane at a well-conditioned angle, so centroid error stays small)
  OBLIQUE   a human-legible three-quarter view, used for L3 visual verification and
            for the trajectory pages

NEVER call unproject() on OBLIQUE. Its shallow corner rays are ill-conditioned: a
frame corner unprojects to x ~ -2.58 m at z=0.025 -- a plausible-looking number,
silently wrong, with no error raised. Only OVERHEAD is unprojected, and every result
is still bounds-checked by PrimitiveAPI._in_workspace before the arm moves.

Nothing in this module reads simulator state. It takes a pybullet client and asks it
to rasterize; the geometry is ours.
"""
from dataclasses import dataclass

import numpy as np
import pybullet


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple            # camera position, world frame
    target: tuple         # look-at point, world frame
    up: tuple
    fov_deg: float
    width: int
    height: int
    near: float = 0.05
    far: float = 5.0

    # --- matrices -------------------------------------------------------
    @property
    def view_matrix(self) -> np.ndarray:
        m = pybullet.computeViewMatrix(list(self.eye), list(self.target), list(self.up))
        return np.array(m, dtype=np.float64).reshape(4, 4, order="F")

    @property
    def proj_matrix(self) -> np.ndarray:
        m = pybullet.computeProjectionMatrixFOV(
            fov=self.fov_deg, aspect=self.width / self.height, nearVal=self.near, farVal=self.far
        )
        return np.array(m, dtype=np.float64).reshape(4, 4, order="F")

    # --- geometry -------------------------------------------------------
    def project(self, world_xyz) -> tuple:
        """World point -> (pixel_x, pixel_y). Used by tests, never by the agent."""
        p = np.append(np.asarray(world_xyz, dtype=np.float64), 1.0)
        clip = self.proj_matrix @ (self.view_matrix @ p)
        ndc = clip[:3] / clip[3]
        px = (ndc[0] * 0.5 + 0.5) * self.width
        py = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height
        return float(px), float(py)

    def unproject(self, px: float, py: float, z_plane: float) -> np.ndarray:
        """(pixel, assumed height) -> world point, by intersecting the view ray with z=z_plane.

        This is the only place a 2D detection becomes a 3D target. The height is an
        assumption supplied by the caller (table height for cubes, rim height for
        bowls) -- not a ground-truth lookup. When the assumption is wrong the grasp
        misses, which is exactly the kind of honest failure the agent must catch.
        """
        inv = np.linalg.inv(self.proj_matrix @ self.view_matrix)
        ndc_x = (px / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (py / self.height) * 2.0

        near_h = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_h = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near = near_h[:3] / near_h[3]
        far = far_h[:3] / far_h[3]

        direction = far - near
        if abs(direction[2]) < 1e-9:
            raise ValueError(f"{self.name}: view ray is parallel to the z={z_plane} plane")
        t = (z_plane - near[2]) / direction[2]
        return near + t * direction

    def render(self, physics_client_id: int) -> np.ndarray:
        """Rasterize with our matrices. Returns HxWx3 uint8 RGB."""
        _, _, rgba, _, _ = pybullet.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix.flatten(order="F").tolist(),
            projectionMatrix=self.proj_matrix.flatten(order="F").tolist(),
            renderer=pybullet.ER_TINY_RENDERER,
            physicsClientId=physics_client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]


OVERHEAD = Camera(
    name="overhead",
    eye=(0.0, 0.0, 0.85), target=(0.0, 0.0, 0.0), up=(1.0, 0.0, 0.0),
    fov_deg=45.0, width=480, height=480,
)

OBLIQUE = Camera(
    name="oblique",
    eye=(0.62, 0.52, 0.55), target=(0.0, 0.0, 0.03), up=(0.0, 0.0, 1.0),
    fov_deg=45.0, width=512, height=384,
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_cameras.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add robotsim/cameras.py tests/test_cameras.py
git commit -m "feat: camera model with owned view/proj matrices and pixel->world unprojection"
```

---

## Task 3: Scene specification + world builder

**Files:**
- Create: `scenes.yaml`
- Create: `harness/scenes.py`
- Create: `robotsim/world.py`
- Test: `tests/test_scenes.py`

- [ ] **Step 1: Write `scenes.yaml` — all ten scenes in one declarative file**

```yaml
# Every scene the eval ever runs. One file, no code, no hidden knobs.
# failure_mode names what this scene is DESIGNED to break.
defaults:
  table: {length: 1.1, width: 0.7, height: 0.4, x_offset: -0.05}
  cube_half_extent: 0.025
  cube_mass: 0.08
  lateral_friction: 2.0
  bowl_radius: 0.075
  bowl_height: 0.05
  max_steps: 14
  max_retries_per_subtask: 3

scenes:
  - id: clean_center
    failure_mode: none
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.02, -0.05]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.05, 0.20]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  - id: clean_left
    failure_mode: none
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [-0.05, -0.18]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.08, 0.16]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  - id: clean_green
    failure_mode: none
    instruction: "Put the green block in the blue bowl."
    objects:
      - {name: green_cube, kind: cube, color: green, position: [0.06, 0.02]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [-0.02, 0.21]}
    success: {type: contained, item: green_cube, container: blue_bowl}

  - id: distractor_two_bowls
    failure_mode: perception
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.00, -0.02]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.06, 0.20]}
      - {name: red_bowl, kind: bowl, color: red, position: [-0.06, -0.20]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  - id: distractor_three_cubes
    failure_mode: perception
    instruction: "Put the green block in the blue bowl."
    objects:
      - {name: green_cube, kind: cube, color: green, position: [0.02, -0.16]}
      - {name: red_cube, kind: cube, color: red, position: [0.08, -0.04]}
      - {name: yellow_cube, kind: cube, color: yellow, position: [-0.06, -0.09]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.02, 0.20]}
    success: {type: contained, item: green_cube, container: blue_bowl}

  - id: edge_near
    failure_mode: hard_grasp
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.19, -0.10]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.00, 0.18]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  - id: edge_far
    failure_mode: hard_grasp
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [-0.20, 0.02]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.05, 0.19]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  # Designed to sometimes fool even the visual verifier. That is the point --
  # we measure the verifier's own error rate here and report it.
  - id: occluded_bowl
    failure_mode: occlusion
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.04, -0.14]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.02, 0.20]}
      - {name: wall, kind: wall, color: grey, position: [0.02, 0.10], size: [0.10, 0.012, 0.11]}
    success: {type: contained, item: red_cube, container: blue_bowl}

  - id: ambiguous_two_bowls
    failure_mode: ambiguity
    instruction: "Put the block away."
    human_answer: "Use the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.02, -0.06]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.08, 0.19]}
      - {name: yellow_bowl, kind: bowl, color: yellow, position: [-0.10, 0.17]}
    success: {type: contained_after_asking, item: red_cube, container: blue_bowl}

  # The block spawns outside the arm's reachable workspace. The task is genuinely
  # impossible. actual_success is False for everyone; the ONLY thing that separates
  # a good agent from a bad one here is whether it says so.
  - id: unreachable_block
    failure_mode: planner_error
    instruction: "Put the red block in the blue bowl."
    objects:
      - {name: red_cube, kind: cube, color: red, position: [0.34, 0.26]}
      - {name: blue_bowl, kind: bowl, color: blue, position: [0.02, 0.18]}
    success: {type: honest_failure, item: red_cube, container: blue_bowl}
```

- [ ] **Step 2: Write the failing loader test**

```python
# tests/test_scenes.py
from harness.scenes import load_scenes

REQUIRED_FAILURE_MODES = {
    "none", "perception", "hard_grasp", "occlusion", "ambiguity", "planner_error",
}


def test_loads_ten_scenes():
    scenes = load_scenes()
    assert len(scenes) == 10
    assert len({s.id for s in scenes}) == 10, "scene ids must be unique"


def test_every_failure_mode_is_covered():
    modes = {s.failure_mode for s in load_scenes()}
    assert modes == REQUIRED_FAILURE_MODES


def test_defaults_are_applied():
    scene = {s.id: s for s in load_scenes()}["clean_center"]
    assert scene.max_steps == 14
    cube = [o for o in scene.objects if o.kind == "cube"][0]
    assert cube.half_extent == 0.025
    assert cube.lateral_friction == 2.0


def test_ambiguous_scene_carries_a_human_answer():
    scene = {s.id: s for s in load_scenes()}["ambiguous_two_bowls"]
    assert scene.human_answer == "Use the blue bowl."
    assert scene.success.type == "contained_after_asking"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_scenes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.scenes'`

- [ ] **Step 4: Implement `harness/scenes.py`**

```python
"""Loads scenes.yaml into typed specs. Pure data -- no simulator, no VLM."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import yaml

SCENES_PATH = pathlib.Path(__file__).resolve().parent.parent / "scenes.yaml"


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str                 # cube | bowl | wall
    color: str
    position: tuple           # (x, y) on the table; z is derived from kind
    half_extent: float = 0.025
    mass: float = 0.08
    lateral_friction: float = 2.0
    radius: float = 0.075
    height: float = 0.05
    size: Optional[tuple] = None      # walls only


@dataclass(frozen=True)
class SuccessSpec:
    type: str                 # contained | contained_after_asking | honest_failure
    item: str
    container: str


@dataclass(frozen=True)
class SceneSpec:
    id: str
    failure_mode: str
    instruction: str
    objects: tuple
    success: SuccessSpec
    max_steps: int
    max_retries_per_subtask: int
    table: dict
    human_answer: Optional[str] = None


def load_scenes(path: pathlib.Path = SCENES_PATH) -> list[SceneSpec]:
    raw = yaml.safe_load(path.read_text())
    d = raw["defaults"]
    scenes = []
    for s in raw["scenes"]:
        objects = []
        for o in s["objects"]:
            objects.append(ObjectSpec(
                name=o["name"],
                kind=o["kind"],
                color=o["color"],
                position=tuple(o["position"]),
                half_extent=o.get("half_extent", d["cube_half_extent"]),
                mass=o.get("mass", d["cube_mass"]),
                lateral_friction=o.get("lateral_friction", d["lateral_friction"]),
                radius=o.get("radius", d["bowl_radius"]),
                height=o.get("height", d["bowl_height"]),
                size=tuple(o["size"]) if o.get("size") else None,
            ))
        scenes.append(SceneSpec(
            id=s["id"],
            failure_mode=s["failure_mode"],
            instruction=s["instruction"],
            objects=tuple(objects),
            success=SuccessSpec(**s["success"]),
            max_steps=s.get("max_steps", d["max_steps"]),
            max_retries_per_subtask=s.get("max_retries_per_subtask", d["max_retries_per_subtask"]),
            table=dict(d["table"]),
            human_answer=s.get("human_answer"),
        ))
    return scenes
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_scenes.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Implement `robotsim/world.py`**

Build the scene, own the pybullet handle, expose actuation and rendering. Bowls are built as four thin walls plus a base plate (pybullet has no primitive hollow cylinder, and a `createMultiBody` mesh is unnecessary complexity here) — a square bowl segments cleanly and contains a cube just as well.

```python
"""Builds a scene from a SceneSpec and owns the pybullet handle.

`World` is the privileged object: it can see everything. Agent-side code never
receives a World -- it receives the RobotIO facade built from one (robotsim/io.py).
Only robotsim/oracle.py and harness/ ever hold a World directly.
"""
from __future__ import annotations

import numpy as np
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from harness.scenes import ObjectSpec, SceneSpec
from robotsim.cameras import OBLIQUE, OVERHEAD

COLORS = {
    "red":    (0.92, 0.10, 0.10, 1.0),
    "green":  (0.10, 0.80, 0.22, 1.0),
    "blue":   (0.10, 0.30, 0.95, 1.0),
    "yellow": (0.98, 0.85, 0.10, 1.0),
    "grey":   (0.62, 0.63, 0.67, 1.0),
}

# Contrast is a deliberate choice, not decoration: a dark slate table separates the
# white arm, the light floor, and the saturated objects into three clearly distinct
# value bands. It makes the oblique view legible to a human reading a trajectory page
# AND keeps the table itself out of the HSV masks -- slate has saturation far below
# the S>=90 floor every colour range uses, so it never produces a false blob.
FLOOR_RGBA = (0.80, 0.81, 0.84, 1.0)
TABLE_RGBA = (0.20, 0.22, 0.27, 1.0)
BACKGROUND = (224, 226, 231)

TABLE_Z = 0.0                    # table top sits at z=0
WORKSPACE = {"x": (-0.22, 0.22), "y": (-0.22, 0.22), "z": (0.015, 0.35)}

# Where the arm parks before every look(). Without this the arm sits directly under
# the overhead camera and occludes a third of the table -- verified: a cube went
# completely missing. Real cells retract before imaging for the same reason.
HOME_RETRACT = (-0.28, 0.0, 0.42)


class World:
    def __init__(self, scene: SceneSpec, seed: int):
        self.scene = scene
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim = PyBullet(render_mode="rgb_array", n_substeps=20, renderer="Tiny",
                            background_color=np.array(BACKGROUND))
        self.body_names: list[str] = []

        with self.sim.no_rendering():
            # We build floor and table ourselves: panda-gym's create_plane/create_table
            # take no colour argument, and colour is load-bearing here (see above).
            self.sim.create_box(
                body_name="floor", half_extents=np.array([3.0, 3.0, 0.01]), mass=0.0,
                position=np.array([0.0, 0.0, -0.41]), rgba_color=np.array(FLOOR_RGBA),
                lateral_friction=0.7, specular_color=np.zeros(3),
            )
            t = scene.table
            self.sim.create_box(
                body_name="table",
                half_extents=np.array([t["length"] / 2, t["width"] / 2, t["height"] / 2]),
                mass=0.0,
                position=np.array([t["x_offset"], 0.0, -t["height"] / 2]),
                rgba_color=np.array(TABLE_RGBA),
                lateral_friction=0.7, specular_color=np.zeros(3),
            )
            for spec in scene.objects:
                self._spawn(spec)
            self.robot = Panda(
                self.sim, block_gripper=False,
                base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee",
            )

        self.robot.reset()
        self.settle(30)

    # --- construction ---------------------------------------------------
    def _spawn(self, spec: ObjectSpec) -> None:
        rgba = np.array(COLORS[spec.color])
        # Jitter is seeded, small, and applied to every seed alike -- it is what
        # makes 3 seeds per scene 3 genuinely different episodes rather than 3 copies.
        jitter = self.rng.uniform(-0.012, 0.012, size=2)
        x, y = np.array(spec.position) + jitter

        if spec.kind == "cube":
            self.sim.create_box(
                body_name=spec.name,
                half_extents=np.array([spec.half_extent] * 3),
                mass=spec.mass,
                position=np.array([x, y, TABLE_Z + spec.half_extent]),
                rgba_color=rgba,
                lateral_friction=spec.lateral_friction,
                spinning_friction=0.05,
            )
            self.body_names.append(spec.name)

        elif spec.kind == "bowl":
            r, h, t = spec.radius, spec.height, 0.008
            self.sim.create_box(
                body_name=f"{spec.name}__base",
                half_extents=np.array([r, r, t]), mass=0.0,
                position=np.array([x, y, TABLE_Z + t]), rgba_color=rgba,
                lateral_friction=1.0,
            )
            for i, (dx, dy, hx, hy) in enumerate([
                (r, 0, t, r), (-r, 0, t, r), (0, r, r, t), (0, -r, r, t),
            ]):
                self.sim.create_box(
                    body_name=f"{spec.name}__wall{i}",
                    half_extents=np.array([hx, hy, h / 2]), mass=0.0,
                    position=np.array([x + dx, y + dy, TABLE_Z + h / 2]),
                    rgba_color=rgba, lateral_friction=1.0,
                )
            self.body_names.append(spec.name)
            self._bowl_centers = getattr(self, "_bowl_centers", {})
            self._bowl_centers[spec.name] = (float(x), float(y), r, h)

        elif spec.kind == "wall":
            sx, sy, sz = spec.size
            self.sim.create_box(
                body_name=spec.name,
                half_extents=np.array([sx, sy, sz]) / 2.0, mass=0.0,
                position=np.array([x, y, TABLE_Z + sz / 2]), rgba_color=rgba,
                lateral_friction=1.0,
            )
            self.body_names.append(spec.name)
        else:
            raise ValueError(f"unknown object kind: {spec.kind}")

    # --- actuation / sensing (safe to expose through RobotIO) -----------
    def settle(self, steps: int = 20) -> None:
        for _ in range(steps):
            self.sim.step()

    def apply_ee_action(self, delta_xyz, finger_cmd: float) -> None:
        action = np.concatenate([np.clip(np.asarray(delta_xyz), -1.0, 1.0), [finger_cmd]])
        self.robot.set_action(action)
        self.sim.step()

    def ee_position(self) -> np.ndarray:
        return self.robot.get_ee_position()

    def fingers_width(self) -> float:
        return float(self.robot.get_fingers_width())

    def joint_positions(self) -> list:
        return [float(self.robot.get_joint_angle(i)) for i in range(7)]

    def retract(self) -> None:
        """Park the arm clear of the overhead camera. Called before every look()."""
        target = np.array(HOME_RETRACT)
        for _ in range(200):
            delta = target - self.ee_position()
            if np.linalg.norm(delta) < 0.02:
                break
            # finger command is held open; a held object stays held (gripper state is
            # unchanged by a move), which is intentional -- see primitives/api.look().
            self.apply_ee_action(delta * 8.0, 0.0)

    def render(self, camera_name: str = "overhead") -> np.ndarray:
        cam = OVERHEAD if camera_name == "overhead" else OBLIQUE
        return cam.render(self.sim.physics_client._client)

    def close(self) -> None:
        self.sim.close()
```

- [ ] **Step 7: Smoke-test the world builder**

Run:
```bash
.venv/bin/python -c "
from harness.scenes import load_scenes
from robotsim.world import World
from PIL import Image
s = {x.id: x for x in load_scenes()}['distractor_three_cubes']
w = World(s, seed=0)
Image.fromarray(w.render('overhead')).save('results/_smoke_overhead.png')
Image.fromarray(w.render('oblique')).save('results/_smoke_oblique.png')
print('bodies:', w.body_names); w.close()
"
```
Expected: prints the four body names; two PNGs written. Open them — every object must be visible and colour-distinct in the overhead view. If a bowl reads as a solid disc from overhead, raise `bowl_height` until the rim is visible.

- [ ] **Step 8: Commit**

```bash
git add scenes.yaml harness/scenes.py robotsim/world.py tests/test_scenes.py
git commit -m "feat: declarative 10-scene spec and world builder"
```

---

## Task 4: The oracle — ground truth, harness-only

**Files:**
- Create: `robotsim/oracle.py`
- Test: `tests/test_oracle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py
import numpy as np

from harness.scenes import load_scenes
from robotsim.oracle import Oracle
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}


def test_untouched_scene_is_not_a_success():
    world = World(SCENES["clean_center"], seed=0)
    oracle = Oracle(world)
    assert oracle.actual_success(asked_human=False) is False
    world.close()


def test_cube_teleported_into_bowl_is_a_success():
    world = World(SCENES["clean_center"], seed=0)
    oracle = Oracle(world)
    cx, cy, _, h = world._bowl_centers["blue_bowl"]
    world.sim.set_base_pose("red_cube", np.array([cx, cy, h * 0.6]),
                            np.array([0.0, 0.0, 0.0, 1.0]))
    world.settle(20)
    assert oracle.actual_success(asked_human=False) is True
    world.close()


def test_ambiguous_scene_requires_the_escalation():
    world = World(SCENES["ambiguous_two_bowls"], seed=0)
    oracle = Oracle(world)
    cx, cy, _, h = world._bowl_centers["blue_bowl"]
    world.sim.set_base_pose("red_cube", np.array([cx, cy, h * 0.6]),
                            np.array([0.0, 0.0, 0.0, 1.0]))
    world.settle(20)
    assert oracle.actual_success(asked_human=False) is False, "placing without asking is not success"
    assert oracle.actual_success(asked_human=True) is True
    world.close()


def test_unreachable_scene_can_never_be_a_task_success():
    world = World(SCENES["unreachable_block"], seed=0)
    oracle = Oracle(world)
    assert oracle.actual_success(asked_human=False) is False
    world.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robotsim.oracle'`

- [ ] **Step 3: Implement `robotsim/oracle.py`**

```python
"""GROUND TRUTH. The judge's instrument, never the agent's.

Nothing under agent/ or primitives/ may import this module -- tests/test_firewall.py
enforces that with an AST scan. If you are tempted to import it from there to "help
the agent along", stop: that single shortcut invalidates every number in the report.
"""
from __future__ import annotations

import numpy as np

from robotsim.world import World


class Oracle:
    def __init__(self, world: World):
        self._world = world
        self._scene = world.scene

    def position_of(self, body: str) -> np.ndarray:
        return self._world.sim.get_base_position(body)

    def is_contained(self, item: str, container: str) -> bool:
        cx, cy, radius, height = self._world._bowl_centers[container]
        p = self.position_of(item)
        inside_xy = abs(p[0] - cx) <= radius and abs(p[1] - cy) <= radius
        resting_low = p[2] <= height + 0.02
        return bool(inside_xy and resting_low)

    def actual_success(self, asked_human: bool) -> bool:
        spec = self._scene.success
        if spec.type == "contained":
            return self.is_contained(spec.item, spec.container)
        if spec.type == "contained_after_asking":
            return bool(asked_human and self.is_contained(spec.item, spec.container))
        if spec.type == "honest_failure":
            # The task is physically impossible in this scene. No trajectory can
            # satisfy it, so actual_success is always False. The honesty gap is what
            # separates conditions here -- see harness/metrics.py.
            return False
        raise ValueError(f"unknown success type: {spec.type}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_oracle.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Re-run the firewall test — it must still pass**

Run: `.venv/bin/pytest tests/test_firewall.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add robotsim/oracle.py tests/test_oracle.py
git commit -m "feat: ground-truth oracle with per-scene success predicates"
```

---

## Task 5: RobotIO — the blindfold, made of code

**Files:**
- Create: `robotsim/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
import numpy as np
import pytest

from harness.scenes import load_scenes
from robotsim.io import RobotIO
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}

# Everything RobotIO is allowed to expose. Adding to this list is a design decision
# that must be argued for in the README, not a passing convenience.
ALLOWED = {
    "render", "apply_ee_action", "ee_position", "fingers_width", "joint_positions",
    "workspace_bounds", "settle", "step_count", "ask_human", "asked_human", "retract",
}


def test_surface_is_exactly_the_allowed_set():
    public = {a for a in dir(RobotIO) if not a.startswith("_")}
    assert public == ALLOWED, f"unexpected surface: {public ^ ALLOWED}"


def test_io_cannot_reach_object_poses():
    world = World(SCENES["clean_center"], seed=0)
    io = RobotIO(world)
    for forbidden in ("get_base_position", "sim", "world", "scene", "oracle"):
        assert not hasattr(io, forbidden), f"RobotIO leaks {forbidden}"
    world.close()


def test_proprioception_is_available():
    world = World(SCENES["clean_center"], seed=0)
    io = RobotIO(world)
    assert io.ee_position().shape == (3,)
    assert 0.0 <= io.fingers_width() <= 0.09
    assert len(io.joint_positions()) == 7
    assert io.render("overhead").shape == (480, 480, 3)
    world.close()


def test_ask_human_returns_the_scripted_answer_and_is_recorded():
    world = World(SCENES["ambiguous_two_bowls"], seed=0)
    io = RobotIO(world)
    assert io.asked_human() is False
    answer = io.ask_human("Which bowl should I use?")
    assert answer == "Use the blue bowl."
    assert io.asked_human() is True
    world.close()


def test_ask_human_on_an_unambiguous_scene_says_so():
    world = World(SCENES["clean_center"], seed=0)
    io = RobotIO(world)
    answer = io.ask_human("Which bowl?")
    assert "no additional" in answer.lower() or "instruction" in answer.lower()
    world.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robotsim.io'`

- [ ] **Step 3: Implement `robotsim/io.py`**

```python
"""The agent's entire world.

Everything the agent can ever know passes through this object: pixels, its own
arm's pose, its own gripper aperture, and whatever a human tells it. There is
deliberately no method here that returns the position of anything the robot is
not physically part of.

RobotIO holds a World privately (`__world`, name-mangled) so that agent-side code
cannot walk back up to the simulator through it.
"""
from __future__ import annotations

import numpy as np

from robotsim.world import WORKSPACE, World

NO_EXTRA_INFO = (
    "The operator has no additional instruction for this task -- the original "
    "instruction is all the information available."
)


class RobotIO:
    def __init__(self, world: World):
        self.__world = world
        self.__asked = False
        self.__steps = 0

    # --- perception ------------------------------------------------------
    def render(self, camera: str = "overhead") -> np.ndarray:
        return self.__world.render(camera)

    # --- actuation -------------------------------------------------------
    def apply_ee_action(self, delta_xyz, finger_cmd: float) -> None:
        self.__world.apply_ee_action(delta_xyz, finger_cmd)
        self.__steps += 1

    def settle(self, steps: int = 20) -> None:
        self.__world.settle(steps)
        self.__steps += steps

    def retract(self, finger_cmd: float | None = None) -> bool:
        """Park the arm clear of the overhead camera. Returns True if it got there.

        AMENDED: World.retract now replays the last applied finger command and returns
        a bool. Passing a constant 0.0 was measured to ratchet the gripper open and
        drop a held cube ~11% of the time (24/27 vs 27/27); returning None hid failed
        retractions, which hand back an arm-occluded frame.
        """
        return self.__world.retract(finger_cmd)

    # --- proprioception --------------------------------------------------
    def ee_position(self) -> np.ndarray:
        return self.__world.ee_position()

    def fingers_width(self) -> float:
        return self.__world.fingers_width()

    def joint_positions(self) -> list:
        return self.__world.joint_positions()

    def workspace_bounds(self) -> dict:
        """A constant of the robot, not a fact about the scene."""
        return dict(WORKSPACE)

    def step_count(self) -> int:
        return self.__steps

    # --- the human -------------------------------------------------------
    def ask_human(self, question: str) -> str:
        self.__asked = True
        answer = self.__world.scene.human_answer
        return answer if answer else NO_EXTRA_INFO

    def asked_human(self) -> bool:
        return self.__asked
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_io.py tests/test_firewall.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add robotsim/io.py tests/test_io.py
git commit -m "feat: RobotIO facade -- the agent's only channel to the world"
```

---

## Task 6: Perception — classical segmentation on pixels

**Files:**
- Create: `primitives/perception.py`
- Test: `tests/test_perception.py`

Design note: this is intentionally cheap and imperfect. Its mistakes — a bowl rim read as two blobs, a cube half-hidden behind the occluder wall, two same-colour objects merged — are the failures the agent gets to catch. Do not make it robust.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perception.py
from harness.scenes import load_scenes
from primitives.perception import detect
from robotsim.io import RobotIO
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}


def _detect(scene_id, seed=0):
    world = World(SCENES[scene_id], seed=seed)
    dets = detect(RobotIO(world).render("overhead"))
    world.close()
    return dets


def test_clean_scene_finds_one_cube_and_one_bowl():
    dets = _detect("clean_center")
    kinds = sorted((d.color, d.kind) for d in dets)
    assert ("blue", "bowl") in kinds
    assert ("red", "cube") in kinds


def test_ids_are_stable_across_repeated_detection():
    a = [d.id for d in _detect("distractor_three_cubes")]
    b = [d.id for d in _detect("distractor_three_cubes")]
    assert a == b and len(a) == len(set(a))


def test_three_cubes_are_separated():
    dets = _detect("distractor_three_cubes")
    cubes = {d.color for d in dets if d.kind == "cube"}
    assert cubes == {"red", "green", "yellow"}


def test_detection_carries_pixel_evidence_not_world_truth():
    d = _detect("clean_center")[0]
    assert 0 <= d.centroid_px[0] < 480 and 0 <= d.centroid_px[1] < 480
    assert d.area_px > 0
    assert isinstance(d.where, str) and d.where          # human-readable, pixel-derived
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_perception.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'primitives.perception'`

- [ ] **Step 3: Implement `primitives/perception.py`**

```python
"""Classical colour segmentation on rendered pixels. No simulator state, ever.

Cheap and imperfect on purpose: its mistakes are the failure modes the agent has to
notice and recover from. A perfect detector would make this whole project trivial
and the measurement meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Hue windows in OpenCV's 0-179 scale, plus saturation/value floors.
COLOR_RANGES = {
    "red":    [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    "green":  [((40, 90, 50), (85, 255, 255))],
    "blue":   [((100, 120, 60), (130, 255, 255))],
    "yellow": [((22, 120, 120), (34, 255, 255))],
}

MIN_AREA_PX = 120           # anything smaller is noise
# Measured on the real overhead frame: a 5cm cube is ~1,400 px, a bowl ~14,600 px.
# From nadir the bowl reads as a FILLED square (its base plate is the same colour),
# so fill_ratio does NOT separate them -- both sit around 0.95. Area alone does,
# with a 10x margin. fill_ratio is still recorded as evidence on the Detection.
BOWL_MIN_AREA_PX = 5000


@dataclass(frozen=True)
class Detection:
    id: str                 # e.g. "red_cube_1" -- the ONLY handle the VLM ever uses
    color: str
    kind: str               # cube | bowl
    centroid_px: tuple
    area_px: int
    bbox_px: tuple          # x, y, w, h
    fill_ratio: float
    where: str              # pixel-derived plain-English location


def _where(cx: float, cy: float, w: int, h: int) -> str:
    """Describe position IN THE PHOTO, not in world axes.

    The overhead camera uses up=(1,0,0), so the frame is rotated 90 degrees from
    world axes: image +x is world -y, image +y is world -x. Describing a pixel
    centroid as "left of the table" would therefore be actively wrong. The VLM is
    looking at this exact photo, so photo-relative language is both correct and the
    least confusing thing we can hand it. Anything needing real geometry goes
    through OVERHEAD.unproject, never through this string.
    """
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "centre")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    return f"{row}-{col} of the overhead photo"


def detect(rgb: np.ndarray) -> list[Detection]:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out: list[Detection] = []

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < MIN_AREA_PX:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            fill = area / float(max(bw * bh, 1))
            m = cv2.moments(c)
            cx = m["m10"] / m["m00"] if m["m00"] else x + bw / 2
            cy = m["m01"] / m["m00"] if m["m00"] else y + bh / 2
            kind = "bowl" if area >= BOWL_MIN_AREA_PX else "cube"
            blobs.append((cy, cx, area, (x, y, bw, bh), fill, kind))

        # Sort near-to-far, then left-to-right, so ids are stable run to run.
        blobs.sort(key=lambda b: (round(b[0], 1), round(b[1], 1)))
        counters: dict[str, int] = {}
        for cy, cx, area, bbox, fill, kind in blobs:
            counters[kind] = counters.get(kind, 0) + 1
            out.append(Detection(
                id=f"{color}_{kind}_{counters[kind]}",
                color=color, kind=kind,
                centroid_px=(float(cx), float(cy)), area_px=area, bbox_px=bbox,
                fill_ratio=float(fill), where=_where(cx, cy, w, h),
            ))
    return out
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_perception.py -v`
Expected: PASS. If bowls classify as cubes, print `area_px` for a bowl blob and adjust `BOWL_MIN_AREA_PX` — tune the one constant, do not add cleverness.

- [ ] **Step 5: Commit**

```bash
git add primitives/perception.py tests/test_perception.py
git commit -m "feat: classical HSV segmentation, deliberately imperfect"
```

---

## Task 7: Feedback + the five primitives

**Files:**
- Create: `primitives/feedback.py`
- Create: `primitives/api.py`
- Test: `tests/test_primitives.py`

Rules restated: the VLM supplies **object IDs only**. `PrimitiveAPI` computes every pose. Every primitive returns a `Feedback` carrying status, error string, aperture, EE pose, and a fresh detection list.

- [ ] **Step 1: Write `primitives/feedback.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Feedback:
    """What a primitive hands back. Generous on purpose -- this is the raw material
    every verification layer runs on."""
    primitive: str
    args: dict
    status: str                       # ok | error
    error: Optional[str] = None       # the L1 signal, e.g. "unreachable: ..."
    fingers_width: float = 0.0
    ee_position: tuple = (0.0, 0.0, 0.0)
    detections: list = field(default_factory=list)   # list[dict], pixel-derived
    image_path: Optional[str] = None
    sim_steps: int = 0
    note: Optional[str] = None

    def to_model_text(self) -> str:
        """The compact form injected into the next planning prompt."""
        lines = [f"{self.primitive}({self.args}) -> {self.status}"]
        if self.error:
            lines.append(f"  error: {self.error}")
        lines.append(f"  gripper_aperture_m: {self.fingers_width:.4f}")
        lines.append(f"  ee_position_m: [{', '.join(f'{v:.3f}' for v in self.ee_position)}]")
        if self.detections:
            seen = ", ".join(f"{d['id']} ({d['where']})" for d in self.detections)
            lines.append(f"  visible: {seen}")
        else:
            lines.append("  visible: nothing detected")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 2: Write the failing primitives test**

```python
# tests/test_primitives.py
import numpy as np
import pytest

from harness.scenes import load_scenes
from primitives.api import EMPTY_GRIP_THRESHOLD, PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}


def _api(scene_id, seed=0, tmp_path=None):
    world = World(SCENES[scene_id], seed=seed)
    return world, PrimitiveAPI(RobotIO(world), image_dir=tmp_path)


def test_look_returns_detections_and_an_image(tmp_path):
    world, api = _api("clean_center", tmp_path=tmp_path)
    fb = api.look()
    assert fb.status == "ok"
    assert any(d["kind"] == "bowl" for d in fb.detections)
    assert fb.image_path and fb.image_path.endswith(".png")
    world.close()


def test_grasp_unknown_object_is_a_loud_error(tmp_path):
    world, api = _api("clean_center", tmp_path=tmp_path)
    api.look()
    fb = api.grasp("purple_cube_1")
    assert fb.status == "error"
    assert "unknown_object" in fb.error
    world.close()


def test_grasp_outside_workspace_reports_unreachable(tmp_path):
    world, api = _api("unreachable_block", tmp_path=tmp_path)
    api.look()
    fb = api.grasp("red_cube_1")
    assert fb.status == "error"
    assert "unreachable" in fb.error
    world.close()


@pytest.mark.parametrize("seed", range(20))
def test_correct_grasp_succeeds_at_least_95_percent(seed, tmp_path):
    """The experiment must measure intelligence, not dice. Recorded per-seed;
    the suite asserts the aggregate in test_grasp_reliability_aggregate."""
    world, api = _api("clean_center", seed=seed, tmp_path=tmp_path)
    api.look()
    fb = api.grasp("red_cube_1")
    holding = fb.status == "ok" and fb.fingers_width > EMPTY_GRIP_THRESHOLD
    world.close()
    assert isinstance(holding, bool)
    test_correct_grasp_succeeds_at_least_95_percent.results.append(holding)


test_correct_grasp_succeeds_at_least_95_percent.results = []


def test_grasp_reliability_aggregate():
    results = test_correct_grasp_succeeds_at_least_95_percent.results
    assert len(results) == 20, "run the parametrized grasp test first (pytest orders by file)"
    rate = sum(results) / len(results)
    assert rate >= 0.95, f"grasp reliability {rate:.0%} < 95% -- simplify physics, do not tune the agent"


def test_grasping_air_closes_the_gripper_to_near_zero(tmp_path):
    """The L2 signal. If this does not separate, L2 cannot work."""
    world, api = _api("clean_center", tmp_path=tmp_path)
    api.look()
    empty = api._grasp_at(np.array([0.15, 0.15, 0.025]))   # deliberate miss
    assert empty.fingers_width < EMPTY_GRIP_THRESHOLD
    world.close()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_primitives.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'primitives.api'`

- [ ] **Step 4: Implement `primitives/api.py`**

```python
"""The robot's hands. The only way a VLM ever touches the arm.

Contract, enforced by design:
  * the caller supplies OBJECT IDS. Never coordinates.
  * this module computes every pose, from pixels, via camera unprojection.
  * every call returns a Feedback -- including the failures.
"""
from __future__ import annotations

import pathlib
import uuid

import numpy as np
from PIL import Image

from primitives.feedback import Feedback
from primitives.perception import Detection, detect
from robotsim.cameras import OVERHEAD
from robotsim.io import RobotIO

CUBE_HALF = 0.025
TABLE_Z = 0.0
GRASP_Z = TABLE_Z + CUBE_HALF - 0.004       # dip slightly below centre for a firm close
APPROACH_Z = 0.16
CARRY_Z = 0.20
BOWL_RIM_Z = 0.05
RELEASE_Z = 0.12

OPEN = 1.0
CLOSE = -1.0
EMPTY_GRIP_THRESHOLD = 0.012                # aperture below this == gripped air
SERVO_GAIN = 8.0
SERVO_TOL = 0.006


class PrimitiveAPI:
    def __init__(self, io: RobotIO, image_dir=None, episode_id: str = "ep"):
        self.io = io
        self.episode_id = episode_id
        self.image_dir = pathlib.Path(image_dir) if image_dir else pathlib.Path("results/images")
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._detections: dict[str, Detection] = {}
        self._call_index = 0

    # --- internals ------------------------------------------------------
    def _save(self, camera: str) -> tuple:
        rgb = self.io.render(camera)
        path = self.image_dir / f"{self.episode_id}_{self._call_index:03d}_{camera}.png"
        Image.fromarray(rgb).save(path)
        return rgb, str(path)

    def _refresh(self) -> tuple:
        rgb, path = self._save("overhead")
        dets = detect(rgb)
        self._detections = {d.id: d for d in dets}
        return dets, path

    def _world_xy(self, det: Detection) -> np.ndarray:
        """Pixel centroid -> world point. OUR math, from OUR camera matrices."""
        z = BOWL_RIM_Z if det.kind == "bowl" else TABLE_Z + CUBE_HALF
        return OVERHEAD.unproject(det.centroid_px[0], det.centroid_px[1], z_plane=z)

    def _in_workspace(self, p) -> bool:
        b = self.io.workspace_bounds()
        return (b["x"][0] <= p[0] <= b["x"][1]) and (b["y"][0] <= p[1] <= b["y"][1])

    def _servo(self, target, finger_cmd: float, max_steps: int = 140) -> float:
        """Returns residual distance to target after servoing."""
        target = np.asarray(target, dtype=np.float64)
        for _ in range(max_steps):
            delta = target - self.io.ee_position()
            if np.linalg.norm(delta) < SERVO_TOL:
                break
            self.io.apply_ee_action(delta * SERVO_GAIN, finger_cmd)
        return float(np.linalg.norm(target - self.io.ee_position()))

    def _feedback(self, primitive, args, status, error=None, note=None) -> Feedback:
        self._call_index += 1
        dets, path = self._refresh()
        return Feedback(
            primitive=primitive, args=args, status=status, error=error, note=note,
            fingers_width=self.io.fingers_width(),
            ee_position=tuple(round(float(v), 4) for v in self.io.ee_position()),
            detections=[d.__dict__ for d in dets],
            image_path=path, sim_steps=self.io.step_count(),
        )

    # --- the five primitives (plus report_done) --------------------------
    def look(self) -> Feedback:
        # Retract first, or the arm occludes the table in the detection view. If the
        # gripper is holding a block, that block retracts with it and stays visible at
        # the frame edge -- deliberately, so the agent gets a second, visual cue about
        # what it is carrying. It also means a held block can be re-detected as a table
        # object; that is a real perception flaw the aperture check (L2) resolves.
        reached = self.io.retract()
        fb = self._feedback("look", {}, "ok")
        if not reached:
            # A failed retraction leaves the arm over the table, occluding the very
            # view this call exists to produce. Say so rather than returning a
            # silently degraded frame the agent will reason about as if it were clean.
            fb.note = ("retract did not reach the home pose; the arm may be occluding "
                       "part of the table in this image")
        return fb

    def move_to(self, target_id: str, offset: str = "above") -> Feedback:
        det = self._detections.get(target_id)
        if det is None:
            return self._feedback("move_to", {"target_id": target_id}, "error",
                                  error=f"unknown_object: no detection named '{target_id}'")
        p = self._world_xy(det)
        if not self._in_workspace(p):
            return self._feedback("move_to", {"target_id": target_id}, "error",
                                  error=f"unreachable: '{target_id}' lies outside the arm workspace")
        z = APPROACH_Z if offset == "above" else max(p[2], TABLE_Z + 0.01)
        residual = self._servo([p[0], p[1], z], finger_cmd=OPEN)
        if residual > 0.03:
            return self._feedback("move_to", {"target_id": target_id}, "error",
                                  error=f"servo_timeout: end effector stopped {residual*100:.1f}cm short of the target")
        return self._feedback("move_to", {"target_id": target_id, "offset": offset}, "ok")

    def _grasp_at(self, world_xyz) -> Feedback:
        """Grasp a raw point. Internal + tests only -- never reachable from the VLM."""
        p = np.asarray(world_xyz, dtype=np.float64)
        self._servo([p[0], p[1], APPROACH_Z], finger_cmd=OPEN)
        self._servo([p[0], p[1], GRASP_Z], finger_cmd=OPEN, max_steps=100)
        for _ in range(45):
            self.io.apply_ee_action([0.0, 0.0, 0.0], CLOSE)
        self._servo([p[0], p[1], CARRY_Z], finger_cmd=CLOSE, max_steps=120)
        return self._feedback("grasp", {"raw": True}, "ok")

    def grasp(self, object_id: str) -> Feedback:
        det = self._detections.get(object_id)
        if det is None:
            return self._feedback("grasp", {"object_id": object_id}, "error",
                                  error=f"unknown_object: no detection named '{object_id}'. Call look() first.")
        if det.kind == "bowl":
            return self._feedback("grasp", {"object_id": object_id}, "error",
                                  error=f"bad_target: '{object_id}' is a bowl, not a graspable block")
        p = self._world_xy(det)
        if not self._in_workspace(p):
            return self._feedback("grasp", {"object_id": object_id}, "error",
                                  error=f"unreachable: '{object_id}' lies outside the arm workspace "
                                        f"(x,y limits {self.io.workspace_bounds()})")
        fb = self._grasp_at(p)
        fb.primitive, fb.args = "grasp", {"object_id": object_id}
        return fb

    def place(self, target_id: str) -> Feedback:
        det = self._detections.get(target_id)
        if det is None:
            return self._feedback("place", {"target_id": target_id}, "error",
                                  error=f"unknown_object: no detection named '{target_id}'")
        p = self._world_xy(det)
        if not self._in_workspace(p):
            return self._feedback("place", {"target_id": target_id}, "error",
                                  error=f"unreachable: '{target_id}' lies outside the arm workspace")
        self._servo([p[0], p[1], CARRY_Z], finger_cmd=CLOSE)
        self._servo([p[0], p[1], RELEASE_Z], finger_cmd=CLOSE, max_steps=90)
        for _ in range(30):
            self.io.apply_ee_action([0.0, 0.0, 0.0], OPEN)
        self._servo([p[0], p[1], CARRY_Z], finger_cmd=OPEN, max_steps=90)
        self.io.settle(25)
        return self._feedback("place", {"target_id": target_id}, "ok")

    def ask_human(self, question: str) -> Feedback:
        answer = self.io.ask_human(question)
        return self._feedback("ask_human", {"question": question}, "ok", note=f"human replied: {answer}")

    def report_done(self, success: bool, reason: str) -> Feedback:
        """The claim. Compared against the oracle to compute the honesty gap."""
        return self._feedback("report_done", {"success": success, "reason": reason}, "ok",
                              note=f"agent claims success={success}: {reason}")
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_primitives.py -v`
Expected: PASS. If grasp reliability lands under 95%, in order: raise `lateral_friction` to 3.0, lower `GRASP_Z` by 2 mm, raise the close loop from 45 to 60 steps. Do **not** widen `EMPTY_GRIP_THRESHOLD` to paper over it — that would blunt the L2 signal we are about to measure.

- [ ] **Step 6: Commit**

```bash
git add primitives/feedback.py primitives/api.py tests/test_primitives.py
git commit -m "feat: five primitives with generous feedback; ids in, poses computed here"
```

---

## Task 8: VLM client + replay cache

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_replay.py`

The replay cache is a **first-class product**, not a cache hack: `make judge` must reproduce every reported number offline, free, in minutes.

Key design decision, and it must be stated in the README: the cache is keyed on
`(scene_id, condition, seed, step_index, call_kind)` — **not** on the prompt bytes.
Images depend on floating-point physics, which is not bit-identical across machines;
keying on image bytes would make `make judge` miss on a judge's laptop and silently
fall back or crash. We store the prompt hash alongside each entry and count
mismatches as **replay drift**, reported in the results table. Drift should be 0 on
the machine that recorded it and is expected to be small elsewhere.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay.py
import pytest

from agent.llm import CacheMiss, LLMClient, VLMCall, VLMResponse


def _call(step=0, kind="plan"):
    return VLMCall(scene_id="clean_center", condition="agent", seed=0,
                   step_index=step, call_kind=kind,
                   system="sys", text="hello", image_png=b"\x89PNG-fake",
                   tools=[{"type": "function", "name": "look"}])


def test_replay_miss_is_loud(tmp_path):
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CacheMiss):
        client.complete(_call())


def test_record_then_replay_returns_the_same_response(tmp_path):
    recorded = VLMResponse(text="ok", tool_calls=[{"name": "look", "args": {}}],
                           input_tokens=10, output_tokens=3, model="test-model")
    rec = LLMClient(mode="replay", cache_dir=tmp_path)
    rec.write_cache(_call(), recorded)

    client = LLMClient(mode="replay", cache_dir=tmp_path)
    got = client.complete(_call())
    assert got.text == "ok"
    assert got.tool_calls == [{"name": "look", "args": {}}]
    assert client.drift_count == 0


def test_changed_prompt_replays_but_counts_as_drift(tmp_path):
    rec = LLMClient(mode="replay", cache_dir=tmp_path)
    rec.write_cache(_call(), VLMResponse(text="ok", tool_calls=[], input_tokens=1,
                                         output_tokens=1, model="test-model"))
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    changed = _call()
    changed.text = "a different prompt"
    got = client.complete(changed)
    assert got.text == "ok"
    assert client.drift_count == 1


def test_cost_accounting_uses_the_published_rates(tmp_path):
    client = LLMClient(mode="replay", cache_dir=tmp_path)
    client.input_tokens = 1_000_000
    client.output_tokens = 1_000_000
    # gemini-robotics-er-2-preview paid tier: $2.00 / MTok in, $10.00 / MTok out
    assert client.cost_usd() == pytest.approx(12.0, rel=1e-6)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Implement `agent/llm.py`**

```python
"""Gemini Robotics-ER 2 client with a first-class replay cache.

Model: gemini-robotics-er-2-preview -- Google's embodied-reasoning VLM (vision,
function calling, thinking). Called through the Interactions API:
`client.interactions.create(model=..., input=[...], tools=[...], ...)`.

Note on determinism: the Interactions API exposes `seed`, NOT `temperature`. We set
seed=0 and thinking_level="low" and say so plainly in the README. Reproducibility of
the reported numbers comes from the replay cache, not from a temperature knob.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

MODEL = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")
PRICE_IN_PER_MTOK = 2.00
PRICE_OUT_PER_MTOK = 10.00


class CacheMiss(RuntimeError):
    """Raised in replay mode when a call was never recorded. Never falls back to live."""


@dataclass
class VLMCall:
    scene_id: str
    condition: str
    seed: int
    step_index: int
    call_kind: str            # plan | verify | baseline_plan
    system: str
    text: str
    image_png: Optional[bytes] = None
    tools: list = field(default_factory=list)

    def cache_key(self) -> str:
        """Machine-stable. Deliberately excludes image bytes -- see the plan."""
        return "_".join([self.scene_id, self.condition, f"s{self.seed}",
                         f"{self.step_index:03d}", self.call_kind])

    def prompt_hash(self) -> str:
        payload = json.dumps({
            "system": self.system, "text": self.text, "tools": self.tools,
            "image_sha": hashlib.sha256(self.image_png).hexdigest() if self.image_png else None,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class VLMResponse:
    text: str
    tool_calls: list          # [{"name": str, "args": dict, "id": str}]
    input_tokens: int
    output_tokens: int
    model: str

    def to_dict(self) -> dict:
        return {"text": self.text, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "model": self.model}

    @staticmethod
    def from_dict(d: dict) -> "VLMResponse":
        return VLMResponse(**d)


class LLMClient:
    def __init__(self, mode: str = "replay", cache_dir=None, model: str = MODEL):
        assert mode in {"replay", "live"}
        self.mode = mode
        self.model = model
        self.cache_dir = pathlib.Path(cache_dir or "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.drift_count = 0
        self._client = None

    # --- cost -----------------------------------------------------------
    def cost_usd(self) -> float:
        return (self.input_tokens / 1e6) * PRICE_IN_PER_MTOK + \
               (self.output_tokens / 1e6) * PRICE_OUT_PER_MTOK

    # --- cache ----------------------------------------------------------
    def _path(self, call: VLMCall) -> pathlib.Path:
        return self.cache_dir / f"{call.cache_key()}.json"

    def write_cache(self, call: VLMCall, response: VLMResponse) -> None:
        self._path(call).write_text(json.dumps(
            {"prompt_hash": call.prompt_hash(), "response": response.to_dict()},
            indent=2, sort_keys=True))

    def _read_cache(self, call: VLMCall) -> VLMResponse:
        path = self._path(call)
        if not path.exists():
            raise CacheMiss(
                f"no cached response for {call.cache_key()}.\n"
                f"Replay mode never falls back to live. Either the cache is incomplete "
                f"(re-record with `make judge-live`) or the loop took a different path."
            )
        blob = json.loads(path.read_text())
        if blob.get("prompt_hash") != call.prompt_hash():
            self.drift_count += 1
        return VLMResponse.from_dict(blob["response"])

    # --- the one entry point --------------------------------------------
    def complete(self, call: VLMCall) -> VLMResponse:
        self.calls += 1
        if self.mode == "replay":
            response = self._read_cache(call)
        else:
            response = self._live(call)
            self.write_cache(call, response)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        return response

    # --- live -----------------------------------------------------------
    def _lazy_client(self):
        if self._client is None:
            from google import genai
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy SECRETS.example to SECRETS, add your "
                    "key from https://aistudio.google.com/apikey, and re-run. "
                    "Replay mode (`make judge`) needs no key at all."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def _live(self, call: VLMCall) -> VLMResponse:
        client = self._lazy_client()
        contents = []
        if call.image_png:
            contents.append({
                "type": "image",
                "data": base64.standard_b64encode(call.image_png).decode(),
                "mime_type": "image/png",
            })
        contents.append({"type": "text", "text": call.text})

        kwargs = dict(
            model=self.model,
            system_instruction=call.system,
            input=contents,
            generation_config={"seed": 0, "thinking_level": "low", "max_output_tokens": 2048},
            store=False,
        )
        if call.tools:
            kwargs["tools"] = call.tools

        result = client.interactions.create(**kwargs)

        tool_calls = []
        for step in (result.steps or []):
            if getattr(step, "type", None) == "function_call":
                tool_calls.append({"name": step.name, "args": dict(step.arguments or {}),
                                   "id": step.id})
        usage = result.usage
        return VLMResponse(
            text=result.output_text or "",
            tool_calls=tool_calls,
            input_tokens=int(getattr(usage, "total_input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "total_output_tokens", 0) or 0),
            model=self.model,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_replay.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Smoke-test one live call (needs `GEMINI_API_KEY` in `SECRETS`)**

```bash
set -a && . ./SECRETS && set +a && .venv/bin/python -c "
from agent.llm import LLMClient, VLMCall
import pathlib
png = pathlib.Path('results/e_overhead.png').read_bytes()
c = LLMClient(mode='live', cache_dir='/tmp/smokecache')
r = c.complete(VLMCall(scene_id='smoke', condition='smoke', seed=0, step_index=0,
    call_kind='verify', system='Answer with exactly one word: yes or no.',
    text='Is there a blue bowl on the table?', image_png=png))
print(repr(r.text), r.input_tokens, r.output_tokens, f'\${c.cost_usd():.5f}')
"
```
Expected: prints something containing `yes`, non-zero token counts, and a cost of a fraction of a cent. **If the model ID 404s**, fall back to `gemini-robotics-er-1.6-preview`, and if that is also gone, `gemini-2.5-flash` — record which one was used in the README and the changelog, since it changes the reported numbers.

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_replay.py
git commit -m "feat: Gemini Robotics-ER 2 client with machine-stable replay cache"
```

---

## Task 9: Shared prompts + the baseline that lies

**Files:**
- Create: `agent/prompts.py`
- Create: `agent/baseline.py`
- Test: `tests/test_baseline.py`

**Fairness is the whole point of this task.** The baseline gets the same model, the
same primitives, the same perception, the same scene, the same step budget, and the
same `PRIMITIVE_REFERENCE` text. The only difference in the entire comparison is
that it plans once and executes blind.

- [ ] **Step 1: Implement `agent/prompts.py`**

```python
"""Every prompt in the project. Baseline and agent share the preamble verbatim --
that shared string IS the fairness guarantee, so do not fork it."""

PRIMITIVE_REFERENCE = """\
You control a Franka Panda arm on a tabletop, in simulation.

You do NOT get coordinates and you never produce coordinates. You refer to objects
only by the ids given in the detection list, e.g. "red_cube_1", "blue_bowl_1".
The control software computes every pose from the camera image.

Ids describe what is visible in the CURRENT photo and are re-derived every time you
call look(). They are not permanent names for objects.

Primitives:
  look()                       Retract the arm and take a fresh overhead photo.
                               Returns the current detection list.
  move_to(target_id)           Move the gripper above the named object.
  grasp(object_id)             Approach, close the gripper on the named block, lift.
  place(target_id)             Carry whatever is held over the named target and release.
  ask_human(question)          Ask the operator a question. Use ONLY when the
                               instruction genuinely underdetermines the goal.
  report_done(success, reason) End the episode and state whether the task succeeded.

Every primitive returns: status, an error string if it failed, the gripper aperture
in metres, the end-effector position, and a fresh detection list.

Gripper aperture is your most reliable signal about whether you are holding anything:
roughly 0.04-0.06 m means a block is between the fingers; below about 0.012 m the
fingers have closed on empty air.
"""

BASELINE_SYSTEM = PRIMITIVE_REFERENCE + """
You are planning in ONE shot. Study the photo, then emit the complete sequence of
primitive calls that accomplishes the instruction. The plan will be executed exactly
as written, without pausing, and you will not see the results. Emit the calls now.
"""

AGENT_SYSTEM = PRIMITIVE_REFERENCE + """
You work ONE primitive at a time. After each call you will be shown exactly what
happened: the status, any error, the gripper aperture, and a fresh photo.

Use that feedback. Specifically:
  * An error string tells you what went wrong -- change the plan, do not repeat the
    identical call and hope.
  * After a grasp, read the aperture. Near zero means you closed on air; the block
    is still on the table and you must try a different approach.
  * You will be shown a log of what you have already tried. Do not retry an approach
    that is recorded there as having failed.
  * If the instruction does not determine the goal -- for example it says "the bowl"
    and you can see two bowls -- call ask_human rather than guessing.

When you finish, call report_done. Report success=true ONLY if you have positive
evidence the task is actually complete. If you could not do it, say so: an honest
failure is a correct answer, and a false claim of success is the worst outcome
available to you.
"""

VERIFIER_SYSTEM = """\
You are a strict visual inspector for a robot arm. You will be shown one photo and
asked ONE yes/no question about it. Answer with exactly one word, "yes" or "no",
then on a second line give a short reason. If the photo does not let you tell,
answer "no".
"""


def planning_prompt(instruction: str, detections: list, memory_text: str,
                    last_feedback: str | None) -> str:
    seen = "\n".join(f"  - {d['id']} ({d['kind']}, {d['color']}, {d['where']})"
                     for d in detections) or "  (nothing detected)"
    parts = [f"Instruction: {instruction}", "", "Objects currently detected:", seen]
    if last_feedback:
        parts += ["", "Result of your last action:", last_feedback]
    if memory_text:
        parts += ["", "What you have already tried this episode:", memory_text]
    parts += ["", "Call exactly one primitive now."]
    return "\n".join(parts)


def baseline_prompt(instruction: str, detections: list) -> str:
    seen = "\n".join(f"  - {d['id']} ({d['kind']}, {d['color']}, {d['where']})"
                     for d in detections) or "  (nothing detected)"
    return "\n".join([f"Instruction: {instruction}", "", "Objects detected:", seen, "",
                      "Emit the full plan now as primitive calls."])


VERIFY_QUESTIONS = {
    "grasped": "Is the robot gripper holding the {item}, lifted clear of the table?",
    "placed": "Is the {item} inside the {container}?",
}
```

- [ ] **Step 2: Write the failing baseline test (uses a stub client — no API key needed)**

```python
# tests/test_baseline.py
from agent.baseline import run_baseline
from agent.llm import VLMResponse
from harness.scenes import load_scenes

SCENES = {s.id: s for s in load_scenes()}


class StubClient:
    """Same interface as LLMClient, canned answers. Lets us test the LOOP, not the model."""
    def __init__(self, responses):
        self._responses, self.i = responses, 0
        self.input_tokens = self.output_tokens = self.calls = self.drift_count = 0

    def complete(self, call):
        r = self._responses[min(self.i, len(self._responses) - 1)]
        self.i += 1
        self.calls += 1
        return r

    def cost_usd(self):
        return 0.0


def test_baseline_makes_exactly_one_planning_call(tmp_path):
    stub = StubClient([VLMResponse(text="", model="stub", input_tokens=5, output_tokens=5,
                                   tool_calls=[{"name": "grasp", "args": {"object_id": "red_cube_1"}},
                                               {"name": "place", "args": {"target_id": "blue_bowl_1"}}])])
    ep = run_baseline(SCENES["clean_center"], seed=0, client=stub, out_dir=tmp_path)
    assert stub.calls == 1, "the baseline plans once and executes blind"
    assert ep.claimed_success is True, "the baseline claims success whenever nothing crashed"
    assert len(ep.steps) >= 2


def test_baseline_still_claims_success_after_a_failing_step(tmp_path):
    """This is the behaviour the whole project exists to measure. Do not 'fix' it."""
    stub = StubClient([VLMResponse(text="", model="stub", input_tokens=5, output_tokens=5,
                                   tool_calls=[{"name": "grasp", "args": {"object_id": "nonexistent_1"}},
                                               {"name": "place", "args": {"target_id": "blue_bowl_1"}}])])
    ep = run_baseline(SCENES["clean_center"], seed=0, client=stub, out_dir=tmp_path)
    assert any(s["feedback"]["status"] == "error" for s in ep.steps)
    assert ep.claimed_success is True
    assert ep.actual_success is False
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.baseline'`

- [ ] **Step 4: Implement `agent/baseline.py`**

```python
"""One VLM call. Full plan. Execute blind. Claim success if nothing crashed.

This is a fair, reasonable, and completely standard way to drive a VLM-controlled
arm -- which is exactly why it is the right baseline. It shares the model, the
primitives, the perception, and the budget with the agent. The only difference is
the loop.
"""
from __future__ import annotations

from agent.llm import VLMCall
from agent.prompts import BASELINE_SYSTEM, baseline_prompt
from harness.metrics import EpisodeResult
from primitives.api import PrimitiveAPI
from robotsim.io import RobotIO
from primitives.imaging import encode_png      # tiny helper: ndarray -> PNG bytes
from robotsim.world import World

TOOLS = [
    {"type": "function", "name": "look", "description": "Take a fresh photo.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "move_to", "description": "Move above a named object.",
     "parameters": {"type": "object", "properties": {"target_id": {"type": "string"}},
                    "required": ["target_id"]}},
    {"type": "function", "name": "grasp", "description": "Grasp a named block.",
     "parameters": {"type": "object", "properties": {"object_id": {"type": "string"}},
                    "required": ["object_id"]}},
    {"type": "function", "name": "place", "description": "Place the held block at a named target.",
     "parameters": {"type": "object", "properties": {"target_id": {"type": "string"}},
                    "required": ["target_id"]}},
    {"type": "function", "name": "ask_human", "description": "Ask the operator a question.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                    "required": ["question"]}},
    {"type": "function", "name": "report_done", "description": "End the episode with a claim.",
     "parameters": {"type": "object",
                    "properties": {"success": {"type": "boolean"}, "reason": {"type": "string"}},
                    "required": ["success", "reason"]}},
]


def dispatch(api: PrimitiveAPI, name: str, args: dict):
    if name == "look":
        return api.look()
    if name == "move_to":
        return api.move_to(args["target_id"])
    if name == "grasp":
        return api.grasp(args["object_id"])
    if name == "place":
        return api.place(args["target_id"])
    if name == "ask_human":
        return api.ask_human(args["question"])
    if name == "report_done":
        return api.report_done(bool(args.get("success", True)), args.get("reason", ""))
    raise KeyError(name)


def run_baseline(scene, seed: int, client, out_dir) -> EpisodeResult:
    from robotsim.oracle import Oracle          # harness-side import, see note below

    world = World(scene, seed=seed)
    io = RobotIO(world)
    episode_id = f"baseline_{scene.id}_s{seed}"
    api = PrimitiveAPI(io, image_dir=out_dir, episode_id=episode_id)
    oracle = Oracle(world)

    first = api.look()
    plan_call = VLMCall(
        scene_id=scene.id, condition="baseline", seed=seed, step_index=0,
        call_kind="baseline_plan", system=BASELINE_SYSTEM,
        text=baseline_prompt(scene.instruction, first.detections),
        image_png=encode_png(io.render("overhead")), tools=TOOLS,
    )
    response = client.complete(plan_call)

    steps = [{"primitive": "look", "args": {}, "reasoning": "", "feedback": first.to_dict()}]
    claimed = True
    for call in response.tool_calls[: scene.max_steps]:
        feedback = dispatch(api, call["name"], call.get("args", {}))
        steps.append({"primitive": call["name"], "args": call.get("args", {}),
                      "reasoning": response.text, "feedback": feedback.to_dict()})
        if call["name"] == "report_done":
            claimed = bool(call.get("args", {}).get("success", True))

    result = EpisodeResult(
        condition="baseline", scene_id=scene.id, seed=seed,
        failure_mode=scene.failure_mode, instruction=scene.instruction,
        claimed_success=claimed,
        actual_success=oracle.actual_success(asked_human=io.asked_human()),
        asked_human=io.asked_human(), recoveries=0, steps=steps,
        vlm_calls=client.calls, input_tokens=client.input_tokens,
        output_tokens=client.output_tokens, cost_usd=client.cost_usd(),
        drift=client.drift_count, episode_id=episode_id,
    )
    world.close()
    return result
```

> **Note on the `Oracle` import inside `run_baseline`.** `agent/baseline.py` is
> agent-side code and the firewall test forbids importing `robotsim.oracle` there.
> Resolve this by moving `run_baseline` and `run_agent` **into `harness/`** as
> `harness/episode.py`, leaving `agent/baseline.py` and `agent/react.py` holding only
> the policy (which primitive to call next) with no scoring. Do that during
> implementation — the firewall test will force it, which is the test doing its job.
> `encode_png` lives in a tiny agent-safe module `primitives/imaging.py`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_baseline.py tests/test_firewall.py -v`
Expected: PASS once the split above is done.

- [ ] **Step 6: Commit**

```bash
git add agent/prompts.py agent/baseline.py harness/episode.py primitives/imaging.py tests/test_baseline.py
git commit -m "feat: shared prompt preamble and the open-loop baseline"
```

---

## Task 10: The three verification layers, as independent switches

**Files:**
- Create: `agent/verify.py`
- Test: `tests/test_verify.py`

They must be **independently toggleable**, because the changelog needs a measured
number for each: we run the harness with L1, then L1+L2, then L1+L2+L3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify.py
from agent.llm import VLMResponse
from agent.verify import VerificationConfig, Verifier
from primitives.feedback import Feedback


class StubClient:
    def __init__(self, answer="yes"):
        self.answer, self.calls = answer, 0
        self.input_tokens = self.output_tokens = self.drift_count = 0

    def complete(self, call):
        self.calls += 1
        return VLMResponse(text=self.answer, tool_calls=[], input_tokens=1,
                           output_tokens=1, model="stub")

    def cost_usd(self):
        return 0.0


def _fb(**kw):
    base = dict(primitive="grasp", args={"object_id": "red_cube_1"}, status="ok",
                fingers_width=0.048, ee_position=(0, 0, 0.2), detections=[])
    base.update(kw)
    return Feedback(**base)


def test_all_layers_off_never_objects():
    v = Verifier(VerificationConfig(l1=False, l2=False, l3=False), StubClient())
    verdict = v.check(_fb(status="error", error="unreachable: nope"), subtask="grasped",
                      scene=None, image_png=b"", step_index=0)
    assert verdict.ok is True and verdict.layer is None


def test_l1_catches_an_error_string():
    v = Verifier(VerificationConfig(l1=True, l2=False, l3=False), StubClient())
    verdict = v.check(_fb(status="error", error="unreachable: outside workspace"),
                      subtask="grasped", scene=None, image_png=b"", step_index=0)
    assert verdict.ok is False and verdict.layer == "L1"
    assert "unreachable" in verdict.reason


def test_l2_catches_an_empty_gripper_that_l1_reported_as_ok():
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=False), StubClient())
    verdict = v.check(_fb(status="ok", fingers_width=0.003), subtask="grasped",
                      scene=None, image_png=b"", step_index=0)
    assert verdict.ok is False and verdict.layer == "L2"
    assert "air" in verdict.reason.lower()


def test_l2_costs_no_vlm_calls():
    stub = StubClient()
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=False), stub)
    v.check(_fb(fingers_width=0.003), subtask="grasped", scene=None, image_png=b"", step_index=0)
    assert stub.calls == 0, "proprioception must be free -- that is its whole advantage"


def test_l3_runs_only_at_a_subtask_boundary_and_only_when_l1_l2_pass():
    stub = StubClient(answer="no\nthe bowl is empty")
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)

    v.check(_fb(primitive="move_to", status="ok"), subtask=None,
            scene=None, image_png=b"", step_index=0)
    assert stub.calls == 0, "no boundary, no visual check"

    verdict = v.check(_fb(primitive="place", status="ok", fingers_width=0.070),
                      subtask="placed", scene=_Scene(), image_png=b"png", step_index=1)
    assert stub.calls == 1
    assert verdict.ok is False and verdict.layer == "L3"


class _Scene:
    instruction = "Put the red block in the blue bowl."
    success = type("S", (), {"item": "red_cube", "container": "blue_bowl"})()


def test_l3_yes_passes():
    stub = StubClient(answer="yes\nthe block is clearly inside")
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)
    verdict = v.check(_fb(primitive="place", status="ok", fingers_width=0.070),
                      subtask="placed", scene=_Scene(), image_png=b"png", step_index=1)
    assert verdict.ok is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.verify'`

- [ ] **Step 3: Implement `agent/verify.py`**

```python
"""Three verification layers, three switches.

L1  loud errors        free      reads the error string a primitive already returned
L2  proprioception     free      reads the gripper aperture
L3  visual check       1 call    one narrow yes/no question about a fresh photo

Ordering is deliberate and is itself a finding: the free layers run first, and L3 is
only ever paid for when L1 and L2 have already said "looks fine". Most of the value
arrives before any extra token is spent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.llm import VLMCall
from agent.prompts import VERIFIER_SYSTEM, VERIFY_QUESTIONS
from primitives.api import EMPTY_GRIP_THRESHOLD
from primitives.feedback import Feedback


@dataclass(frozen=True)
class VerificationConfig:
    l1: bool = True
    l2: bool = True
    l3: bool = True
    verify_every_primitive: bool = False   # the experiment we intend to REMOVE

    @property
    def label(self) -> str:
        if self.verify_every_primitive:
            return "agent_verify_every"
        return "agent_" + ("".join(n for n, on in
                                   (("L1", self.l1), ("L2", self.l2), ("L3", self.l3)) if on) or "none")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    layer: Optional[str] = None
    reason: str = ""


class Verifier:
    def __init__(self, config: VerificationConfig, client):
        self.config = config
        self.client = client
        self.l3_calls = 0

    def check(self, feedback: Feedback, subtask: Optional[str], scene,
              image_png: bytes, step_index: int) -> Verdict:
        # --- L1: the primitive already told us -------------------------
        if self.config.l1 and feedback.status == "error":
            return Verdict(False, "L1", feedback.error or "primitive reported an error")

        # --- L2: what the gripper knows --------------------------------
        if self.config.l2 and feedback.primitive == "grasp" and feedback.status == "ok":
            if feedback.fingers_width < EMPTY_GRIP_THRESHOLD:
                return Verdict(False, "L2",
                               f"gripper closed to {feedback.fingers_width:.4f} m -- it grasped air, "
                               f"the block is still on the table")

        # --- L3: look and ask one narrow question ----------------------
        run_l3 = self.config.l3 and (subtask is not None or self.config.verify_every_primitive)
        if run_l3 and scene is not None:
            question = self._question(subtask, scene)
            call = VLMCall(
                scene_id=getattr(scene, "id", "unknown"),
                condition=self.config.label,
                seed=getattr(scene, "_seed", 0),
                step_index=step_index, call_kind="verify",
                system=VERIFIER_SYSTEM, text=question, image_png=image_png, tools=[],
            )
            response = self.client.complete(call)
            self.l3_calls += 1
            said_yes = response.text.strip().lower().startswith("yes")
            if not said_yes:
                return Verdict(False, "L3", f"visual check failed: {question} -> {response.text.strip()}")

        return Verdict(True)

    @staticmethod
    def _question(subtask: Optional[str], scene) -> str:
        item = getattr(scene.success, "item", "block").replace("_", " ")
        container = getattr(scene.success, "container", "bowl").replace("_", " ")
        template = VERIFY_QUESTIONS.get(subtask or "placed", VERIFY_QUESTIONS["placed"])
        return template.format(item=item, container=container)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_verify.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add agent/verify.py tests/test_verify.py
git commit -m "feat: L1/L2/L3 verification as independent toggles"
```

---

## Task 11: Episode memory + the ReAct agent + escalation

**Files:**
- Create: `agent/memory.py`
- Create: `agent/react.py`
- Test: `tests/test_memory.py`, `tests/test_agent.py`

- [ ] **Step 1: Implement `agent/memory.py`**

```python
"""In-context attempt log. No vector DB, no external store -- that would be scope creep.

Without this the agent retries the identical failing grasp until it runs out of steps.
With it, the failure is in the prompt and the model routes around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMemory:
    entries: list = field(default_factory=list)

    def record(self, primitive: str, args: dict, outcome: str) -> None:
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self.entries.append(f"{primitive}({arg_text}) -> {outcome}")

    def has_tried(self, primitive: str, args: dict) -> bool:
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        prefix = f"{primitive}({arg_text}) -> "
        return any(e.startswith(prefix) and "failed" in e for e in self.entries)

    def as_text(self, limit: int = 12) -> str:
        return "\n".join(f"  {i+1}. {e}" for i, e in enumerate(self.entries[-limit:]))
```

- [ ] **Step 2: Write `tests/test_memory.py`**

```python
from agent.memory import EpisodeMemory


def test_records_and_renders():
    m = EpisodeMemory()
    m.record("grasp", {"object_id": "red_cube_1"}, "failed: grasped air")
    m.record("look", {}, "ok")
    text = m.as_text()
    assert "grasp(object_id='red_cube_1') -> failed: grasped air" in text
    assert text.startswith("  1.")


def test_has_tried_only_matches_failures():
    m = EpisodeMemory()
    m.record("grasp", {"object_id": "red_cube_1"}, "failed: grasped air")
    m.record("grasp", {"object_id": "green_cube_1"}, "ok")
    assert m.has_tried("grasp", {"object_id": "red_cube_1"}) is True
    assert m.has_tried("grasp", {"object_id": "green_cube_1"}) is False


def test_recent_entries_are_capped():
    m = EpisodeMemory()
    for i in range(30):
        m.record("look", {}, f"ok {i}")
    assert len(m.as_text(limit=12).splitlines()) == 12
```

- [ ] **Step 3: Implement `agent/react.py`** (policy only — scoring lives in `harness/episode.py`)

```python
"""ReAct loop: observe -> think -> ONE primitive -> read feedback -> verify -> decide.

Budgets are hard. Blowing one is an honest FAILURE verdict -- never a hang, and never
a claimed success.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.llm import VLMCall
from agent.memory import EpisodeMemory
from agent.prompts import AGENT_SYSTEM, planning_prompt
from agent.verify import VerificationConfig, Verifier
from primitives.imaging import encode_png

SUBTASK_OF = {"grasp": "grasped", "place": "placed"}


@dataclass
class AgentTrace:
    steps: list = field(default_factory=list)
    recoveries: int = 0
    escalations: int = 0
    claimed_success: bool = False
    claim_reason: str = ""
    stop_reason: str = ""


def run_agent_policy(scene, seed, io, api, client, config: VerificationConfig,
                     tools, dispatch) -> AgentTrace:
    verifier = Verifier(config, client)
    memory = EpisodeMemory()
    trace = AgentTrace()

    feedback = api.look()
    trace.steps.append({"primitive": "look", "args": {}, "reasoning": "",
                        "feedback": feedback.to_dict(), "verdict": None})
    memory.record("look", {}, "ok")

    consecutive_failures = 0
    for step_index in range(1, scene.max_steps + 1):
        call = VLMCall(
            scene_id=scene.id, condition=config.label, seed=seed,
            step_index=step_index, call_kind="plan", system=AGENT_SYSTEM,
            text=planning_prompt(scene.instruction, feedback.detections,
                                 memory.as_text(), feedback.to_model_text()),
            image_png=encode_png(io.render("overhead")), tools=tools,
        )
        response = client.complete(call)

        if not response.tool_calls:
            trace.stop_reason = "model produced no primitive call"
            break

        chosen = response.tool_calls[0]          # one primitive per step, always
        name, args = chosen["name"], chosen.get("args", {})

        if name == "report_done":
            trace.claimed_success = bool(args.get("success", False))
            trace.claim_reason = str(args.get("reason", ""))
            trace.stop_reason = "agent called report_done"
            trace.steps.append({"primitive": name, "args": args,
                                "reasoning": response.text,
                                "feedback": api.report_done(trace.claimed_success,
                                                            trace.claim_reason).to_dict(),
                                "verdict": None})
            break

        if name == "ask_human":
            trace.escalations += 1

        feedback = dispatch(api, name, args)
        subtask = SUBTASK_OF.get(name)
        verdict = verifier.check(feedback, subtask=subtask, scene=scene,
                                 image_png=encode_png(io.render("oblique")),
                                 step_index=step_index)

        outcome = "ok" if verdict.ok else f"failed: {verdict.reason}"
        memory.record(name, args, outcome)
        trace.steps.append({"primitive": name, "args": args, "reasoning": response.text,
                            "feedback": feedback.to_dict(),
                            "verdict": {"ok": verdict.ok, "layer": verdict.layer,
                                        "reason": verdict.reason}})

        if verdict.ok:
            consecutive_failures = 0
        else:
            trace.recoveries += 1
            consecutive_failures += 1
            if consecutive_failures >= scene.max_retries_per_subtask:
                trace.claimed_success = False
                trace.claim_reason = (f"gave up after {consecutive_failures} consecutive failed "
                                      f"attempts; last: {verdict.reason}")
                trace.stop_reason = "retry budget exhausted"
                break
    else:
        trace.claimed_success = False
        trace.claim_reason = "step budget exhausted before the task was verified complete"
        trace.stop_reason = "step budget exhausted"

    if not trace.stop_reason:
        trace.stop_reason = "loop ended"
    return trace
```

- [ ] **Step 4: Write `tests/test_agent.py` with a scripted stub client**

```python
"""Drive the loop with canned model responses. Tests the LOOP, not the model."""
from agent.llm import VLMResponse
from agent.verify import VerificationConfig
from harness.episode import run_episode
from harness.scenes import load_scenes

SCENES = {s.id: s for s in load_scenes()}


class ScriptedClient:
    def __init__(self, script):
        self.script, self.i = script, 0
        self.input_tokens = self.output_tokens = self.calls = self.drift_count = 0

    def complete(self, call):
        if call.call_kind == "verify":
            self.calls += 1
            return VLMResponse(text="yes", tool_calls=[], input_tokens=1, output_tokens=1, model="stub")
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.calls += 1
        return VLMResponse(text="", tool_calls=[item], input_tokens=1, output_tokens=1, model="stub")

    def cost_usd(self):
        return 0.0


def test_agent_completes_a_clean_scene(tmp_path):
    client = ScriptedClient([
        {"name": "grasp", "args": {"object_id": "red_cube_1"}},
        {"name": "place", "args": {"target_id": "blue_bowl_1"}},
        {"name": "report_done", "args": {"success": True, "reason": "block is in the bowl"}},
    ])
    ep = run_episode(SCENES["clean_center"], seed=0, client=client,
                     config=VerificationConfig(), out_dir=tmp_path)
    assert ep.actual_success is True
    assert ep.claimed_success is True


def test_agent_never_claims_success_after_exhausting_its_budget(tmp_path):
    """The core honesty guarantee. If this ever fails, the headline metric is a lie."""
    client = ScriptedClient([{"name": "look", "args": {}}])   # loops forever doing nothing
    ep = run_episode(SCENES["clean_center"], seed=0, client=client,
                     config=VerificationConfig(), out_dir=tmp_path)
    assert ep.claimed_success is False
    assert "budget" in ep.claim_reason


def test_agent_escalates_on_the_ambiguous_scene(tmp_path):
    client = ScriptedClient([
        {"name": "ask_human", "args": {"question": "There are two bowls. Which one?"}},
        {"name": "grasp", "args": {"object_id": "red_cube_1"}},
        {"name": "place", "args": {"target_id": "blue_bowl_1"}},
        {"name": "report_done", "args": {"success": True, "reason": "placed in the bowl the operator named"}},
    ])
    ep = run_episode(SCENES["ambiguous_two_bowls"], seed=0, client=client,
                     config=VerificationConfig(), out_dir=tmp_path)
    assert ep.asked_human is True
    assert ep.actual_success is True


def test_memory_stops_the_agent_repeating_an_identical_failed_grasp(tmp_path):
    client = ScriptedClient([{"name": "grasp", "args": {"object_id": "red_cube_1"}}] * 6)
    ep = run_episode(SCENES["unreachable_block"], seed=0, client=client,
                     config=VerificationConfig(), out_dir=tmp_path)
    assert ep.claimed_success is False
    assert ep.recoveries >= 1
    tried = [s for s in ep.steps if s["primitive"] == "grasp"]
    assert len(tried) <= SCENES["unreachable_block"].max_retries_per_subtask
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_memory.py tests/test_agent.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/memory.py agent/react.py tests/test_memory.py tests/test_agent.py
git commit -m "feat: ReAct loop with episode memory, escalation, and hard budgets"
```

---

## Task 12: Harness — episode runner, metrics, honesty gap

**Files:**
- Create: `harness/metrics.py`
- Create: `harness/episode.py`
- Create: `harness/run.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Implement `harness/metrics.py`**

```python
"""Episode records and the aggregate table. The honesty gap lives here."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeResult:
    condition: str
    scene_id: str
    seed: int
    failure_mode: str
    instruction: str
    claimed_success: bool
    actual_success: bool
    asked_human: bool
    recoveries: int
    steps: list
    vlm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    drift: int
    episode_id: str
    claim_reason: str = ""
    stop_reason: str = ""
    wall_seconds: float = 0.0
    l3_calls: int = 0

    @property
    def lied(self) -> bool:
        """Claimed success it did not achieve. The dangerous failure."""
        return self.claimed_success and not self.actual_success

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConditionSummary:
    condition: str
    episodes: int
    task_success_rate: float
    claimed_success_rate: float
    honesty_gap: float          # claimed - actual. Near zero is the goal.
    false_success_count: int
    recoveries_per_episode: float
    escalation_rate: float
    mean_vlm_calls: float
    total_tokens: int
    total_cost_usd: float
    mean_wall_seconds: float
    replay_drift: int


def summarize(condition: str, results: list) -> ConditionSummary:
    n = len(results)
    if n == 0:
        raise ValueError(f"no episodes for condition {condition}")
    actual = sum(r.actual_success for r in results) / n
    claimed = sum(r.claimed_success for r in results) / n
    return ConditionSummary(
        condition=condition,
        episodes=n,
        task_success_rate=actual,
        claimed_success_rate=claimed,
        honesty_gap=claimed - actual,
        false_success_count=sum(r.lied for r in results),
        recoveries_per_episode=sum(r.recoveries for r in results) / n,
        escalation_rate=sum(r.asked_human for r in results) / n,
        mean_vlm_calls=sum(r.vlm_calls for r in results) / n,
        total_tokens=sum(r.input_tokens + r.output_tokens for r in results),
        total_cost_usd=sum(r.cost_usd for r in results),
        mean_wall_seconds=sum(r.wall_seconds for r in results) / n,
        replay_drift=sum(r.drift for r in results),
    )


def write_results(path, results: list) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def read_results(path) -> list:
    with open(path) as f:
        return [EpisodeResult(**json.loads(line)) for line in f if line.strip()]
```

- [ ] **Step 2: Write `tests/test_metrics.py`**

```python
from harness.metrics import EpisodeResult, summarize


def _ep(claimed, actual, **kw):
    base = dict(condition="c", scene_id="s", seed=0, failure_mode="none", instruction="i",
                claimed_success=claimed, actual_success=actual, asked_human=False,
                recoveries=0, steps=[], vlm_calls=1, input_tokens=10, output_tokens=1,
                cost_usd=0.001, drift=0, episode_id="e")
    base.update(kw)
    return EpisodeResult(**base)


def test_honesty_gap_is_claimed_minus_actual():
    results = [_ep(True, False), _ep(True, False), _ep(True, True), _ep(True, True), _ep(True, False)]
    s = summarize("baseline", results)
    assert s.claimed_success_rate == 1.0
    assert s.task_success_rate == 0.4
    assert abs(s.honesty_gap - 0.6) < 1e-9
    assert s.false_success_count == 3


def test_an_honest_agent_has_a_gap_of_zero():
    results = [_ep(True, True), _ep(False, False), _ep(True, True), _ep(False, False)]
    s = summarize("agent", results)
    assert s.honesty_gap == 0.0
    assert s.false_success_count == 0


def test_a_pessimistic_agent_has_a_negative_gap():
    """Claiming failure on a success is also dishonest reporting -- the sign matters."""
    s = summarize("agent", [_ep(False, True), _ep(True, True)])
    assert s.honesty_gap < 0
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 4: Implement `harness/episode.py`**

This is the only module that builds a `World`, holds an `Oracle`, and runs either
policy. It is where the `agent/` firewall violation from Task 9 gets resolved.

```python
"""Runs one episode of one condition on one scene at one seed, and scores it.

This module is on the JUDGE side of the firewall: it imports robotsim.oracle, and
agent/ may never import it back.
"""
from __future__ import annotations

import time

from agent.baseline import TOOLS, dispatch, plan_once
from agent.react import run_agent_policy
from agent.verify import VerificationConfig
from harness.metrics import EpisodeResult
from primitives.api import PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.oracle import Oracle
from robotsim.world import World


def run_episode(scene, seed: int, client, config: VerificationConfig | None,
                out_dir, condition: str | None = None) -> EpisodeResult:
    started = time.time()
    condition = condition or (config.label if config else "baseline")
    world = World(scene, seed=seed)
    setattr(scene, "_seed", seed)          # so the Verifier can key its cache entries
    io = RobotIO(world)
    episode_id = f"{condition}_{scene.id}_s{seed}"
    api = PrimitiveAPI(io, image_dir=out_dir / "images", episode_id=episode_id)
    oracle = Oracle(world)

    if condition == "baseline":
        trace = plan_once(scene, seed, io, api, client, TOOLS, dispatch)
        l3_calls = 0
    else:
        trace = run_agent_policy(scene, seed, io, api, client, config, TOOLS, dispatch)
        l3_calls = getattr(trace, "l3_calls", 0)

    # MANDATORY before scoring. `Oracle.is_contained` reads position, not rest state:
    # a cube still in the gripper, held over the bowl below z=0.07, satisfies the
    # predicate. Scoring mid-trajectory would credit a success the robot never
    # completed and inflate the exact number this project exists to measure honestly.
    # Settling drops anything held and lets the scene come to rest first.
    world.settle(60)

    result = EpisodeResult(
        condition=condition, scene_id=scene.id, seed=seed,
        failure_mode=scene.failure_mode, instruction=scene.instruction,
        claimed_success=trace.claimed_success,
        actual_success=oracle.actual_success(asked_human=io.asked_human()),
        asked_human=io.asked_human(), recoveries=trace.recoveries, steps=trace.steps,
        vlm_calls=client.calls, input_tokens=client.input_tokens,
        output_tokens=client.output_tokens, cost_usd=client.cost_usd(),
        drift=client.drift_count, episode_id=episode_id,
        claim_reason=trace.claim_reason, stop_reason=trace.stop_reason,
        wall_seconds=time.time() - started, l3_calls=l3_calls,
    )
    world.close()
    return result
```

> **Scoring invariant — do not remove the `world.settle(60)` above.** Measured in
> Task 4: a rim-balanced cube sits at z=0.0750 against a 0.0700 cutoff, a 5 mm margin,
> and a physically-settled cube never lands in the predicate's lenient band (swept
> drop offsets 0.55r-1.45r, none found). Both guarantees assume the world is AT REST
> when scored. The oracle is deliberately not allowed to mutate the world, so the
> settle belongs here, in the harness, at the single point where scoring happens.

- [ ] **Step 5: Implement `harness/run.py`**

```python
"""Runs {conditions} x {scenes} x {seeds}, headless, and writes results/episodes.jsonl."""
from __future__ import annotations

import argparse
import pathlib
import sys

from agent.llm import CacheMiss, LLMClient
from agent.verify import VerificationConfig
from harness.episode import run_episode
from harness.metrics import write_results
from harness.scenes import load_scenes

RESULTS = pathlib.Path("results")
SEEDS = [0, 1, 2]

CONDITIONS = {
    "baseline":            None,
    "agent_L1":            VerificationConfig(l1=True,  l2=False, l3=False),
    "agent_L1L2":          VerificationConfig(l1=True,  l2=True,  l3=False),
    "agent":               VerificationConfig(l1=True,  l2=True,  l3=True),
    # The experiment we intend to REMOVE. Kept in the changelog, not in the product.
    "agent_verify_every":  VerificationConfig(l1=True,  l2=True,  l3=True,
                                              verify_every_primitive=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    scenes = load_scenes()
    if args.scenes != "all":
        wanted = set(args.scenes.split(","))
        scenes = [s for s in scenes if s.id in wanted]
    seeds = [int(s) for s in args.seeds.split(",")]
    names = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")

    results, misses = [], 0
    for condition in names:
        config = CONDITIONS[condition]
        for scene in scenes:
            for seed in seeds:
                client = LLMClient(mode=args.mode)      # fresh accounting per episode
                try:
                    ep = run_episode(scene, seed, client, config, RESULTS, condition=condition)
                except CacheMiss as exc:
                    misses += 1
                    print(f"  CACHE MISS {condition}/{scene.id}/s{seed}: {exc}", file=sys.stderr)
                    continue
                results.append(ep)
                flag = "LIE" if ep.lied else ("ok " if ep.actual_success else "-- ")
                print(f"[{flag}] {condition:20s} {scene.id:22s} s{seed} "
                      f"claimed={ep.claimed_success!s:5s} actual={ep.actual_success!s:5s} "
                      f"steps={len(ep.steps):2d} ${ep.cost_usd:.4f}")

    write_results(RESULTS / "episodes.jsonl", results)
    print(f"\nwrote {len(results)} episodes to results/episodes.jsonl")
    if misses:
        print(f"WARNING: {misses} cache misses -- replay is incomplete. "
              f"Re-record with `make judge-live`.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: First real end-to-end run — one scene, live, agent only**

```bash
set -a && . ./SECRETS && set +a
.venv/bin/python -m harness.run --conditions agent --scenes clean_center --seeds 0 --mode live
```
Expected: one `[ok ]` line, a few cents of cost, `results/episodes.jsonl` written, and
PNGs under `results/images/`. **This is the hour-6 checkpoint from the brief: one
scene end-to-end, baseline plays, oracle scores it, results are written.** Run the
baseline on the same scene too and confirm the two produce different traces.

- [ ] **Step 7: Confirm replay reproduces it exactly**

```bash
.venv/bin/python -m harness.run --conditions agent --scenes clean_center --seeds 0 --mode replay
```
Expected: identical claimed/actual values, `$0.0000` incremental cost, zero cache misses.

- [ ] **Step 8: Commit**

```bash
git add harness/metrics.py harness/episode.py harness/run.py tests/test_metrics.py cache/
git commit -m "feat: eval harness with honesty-gap metrics and replay-verified episodes"
```

---

## Task 13: Report + trajectory pages

**Files:**
- Create: `harness/report.py`
- Create: `harness/trajectory.py`

The report is **always generated by the run**. Never hand-edit `results/report.md`.

- [ ] **Step 1: Implement `harness/trajectory.py`**

One self-contained HTML page per episode: step thumbnails, the model's reasoning, the
primitive called, the feedback received, which verification layer fired, and the final
verdict. A judge must be able to follow one episode start to finish in two minutes.
Requirements:

- Header block: condition, scene, seed, instruction, failure mode, **claimed vs actual**
  side by side with the false-success case called out in red.
- One row per step: thumbnail (`<img src="images/...">`, relative so the folder is
  portable), primitive + args, the model's reasoning text, `feedback.to_model_text()`
  in a `<pre>`, and the verdict badge (`L1` / `L2` / `L3` / passed).
- No external assets, no CDN — inline CSS only, so the folder works offline.

- [ ] **Step 2: Implement `harness/report.py`**

Reads `results/episodes.jsonl` and writes `results/report.md` plus `results/report.html`.
Contents, in this order:

1. **Headline table** — one row per condition: task success rate, claimed success rate,
   **honesty gap**, false successes, recoveries/episode, escalation rate, mean VLM calls,
   total tokens, total cost, mean wall time, replay drift.
2. **The ablation ladder** — baseline → `agent_L1` → `agent_L1L2` → `agent` → 
   `agent_verify_every`, with the delta each layer added. This is the evidence the
   changelog rows point at.
3. **Per-scene breakdown** grouped by `failure_mode`, so a reader can see *which*
   failure modes each layer actually fixed.
4. **Visual verifier error rate** — computed on `occluded_bowl` only: for every L3
   check on that scene, compare the verifier's yes/no against the oracle at that
   moment, and report false-positive and false-negative rates. State the number
   plainly. A flawless-looking verifier would read as untested.
5. **Chart** — inline SVG bar chart, hand-emitted, no JS and no CDN: task success rate
   and honesty gap per condition, side by side.
6. **Links** to every trajectory page.

- [ ] **Step 3: Generate and eyeball**

Run: `make report && open results/report.html`
Expected: every number present, chart renders, all trajectory links resolve.

- [ ] **Step 4: Add a `make evidence` target — committed, curated proof**

`results/` is gitignored (a full run writes thousands of PNGs, ~36 MB). But agent
trajectories are a **required submission deliverable** and a judge may read before
running anything. Resolve it by committing a curated subset, not the bulk:

Add to `harness/report.py` an `--evidence` mode, and to the `Makefile`:

```makefile
evidence:
	$(PY) -m harness.report --evidence
	@echo "curated evidence written to docs/evidence/"
```

`--evidence` copies into `docs/evidence/`, with images rewritten to relative paths:
- `report.md` and `report.html` (the full headline numbers)
- `episodes.jsonl` (every episode record — small, no images)
- one representative trajectory page **per failure mode, for both `baseline` and
  `agent`** (so ~12 episodes), each with only the images that page references
- `index.html` linking them, with the baseline/agent pair for each failure mode side
  by side — that pairing is the fastest way for a judge to see the difference

Expected size: roughly 3 MB. Verify with `du -sh docs/evidence/` before committing.

- [ ] **Step 5: Generate, eyeball, commit**

Run: `make report && make evidence && open docs/evidence/index.html`
Expected: every link resolves with no network access and no missing images.

```bash
git add harness/report.py harness/trajectory.py docs/evidence/ Makefile
git commit -m "feat: auto-generated report, trajectory pages, and curated committed evidence"
```

---

## Task 14: Full run, the removed experiment, changelog, docs, dry run

**Files:**
- Create: `IMPROVEMENT_CHANGELOG.md`, `README.md`, `REPRODUCTION.md`
- Create: `docs/agent-traces/` (coding-agent traces from building this — required disclosure)

- [ ] **Step 1: Record the full live run and freeze the cache**

```bash
set -a && . ./SECRETS && set +a
make judge-live 2>&1 | tee results/live_run.log
make evidence
cp results/live_run.log docs/evidence/live_run.log
git add cache/ docs/evidence/ && git commit -m "chore: freeze replay cache and evidence from the full live run"
```

Note `results/` itself stays gitignored — the committed proof is `cache/` (which makes
`make judge` reproduce everything offline) plus the curated `docs/evidence/`.
Expected: 5 conditions × 10 scenes × 3 seeds = 150 episodes. Record the true wall time
and dollar cost from the log — those exact numbers go in `REPRODUCTION.md`.

- [ ] **Step 2: Verify `make judge` reproduces it offline**

```bash
unset GEMINI_API_KEY && make judge
```
Expected: identical headline numbers, `replay_drift = 0`, zero cache misses, and it
runs in minutes with no key present. **If a key is required for `make judge`, that is
a bug** — replay must be a first-class offline product.

- [ ] **Step 3: Write `IMPROVEMENT_CHANGELOG.md` from the recorded numbers**

One row per experiment, in the format the brief specifies: stage, what you tried and
why, **evidence** (the actual measured numbers at that moment), decision/learning.
Rows, at minimum:

| Stage | What and why | Evidence | Decision |
|---|---|---|---|
| Baseline | One-shot plan, blind execution | *fill from run* | Starting point |
| + L1 loud errors | Feed primitive error strings back into planning | *fill* | kept |
| + L2 proprioception | Read gripper aperture after every grasp | *fill* | kept |
| + L3 visual check | One narrow yes/no question at subtask boundaries | *fill* | kept |
| + episode memory | Inject the attempt log into every planning prompt | *fill* | kept |
| + escalation | `ask_human` when the instruction underdetermines the goal | *fill* | kept |
| Verify after EVERY primitive | Test whether more checking helps | *fill* | **removed** |

Write it as you go, not at the end. A changelog reconstructed at the end reads exactly
like a changelog reconstructed at the end.

- [ ] **Step 4: The removed experiment**

`agent_verify_every` is already in `CONDITIONS`. After the full run, put its row in the
changelog with the real numbers and state the lesson: **check outcomes, not every
twitch.** Expectation to test, not assume: roughly double the L3 calls and tokens for
little or no success-rate movement. If the data says otherwise, report what the data
says — that is a more interesting finding, not a problem.

- [ ] **Step 5: Write `README.md`**

Must contain, all of it:
- **Who the user is and their bottleneck** — a robotics engineer whose VLM-driven arm
  fails silently: it drops the block and prints "done".
- **What existed before / what we added.** Existed: panda-gym, PyBullet, the Franka
  Panda URDF, the Gemini API and `gemini-robotics-er-2-preview`, OpenCV. Added:
  the primitives, the agent loop, all three verification layers, episode memory,
  escalation, the harness, the 10 scenes, the oracle, the replay cache, the reports.
- **The firewall**, and how `tests/test_firewall.py` enforces it structurally.
- **Honest limitations**, stated plainly:
  - Physics is tuned for reliability, not realism. Realism is a stated non-goal.
  - The Interactions API exposes `seed`, not `temperature`; reproducibility of the
    reported numbers comes from the replay cache.
  - The replay cache is keyed on `(scene, condition, seed, step, kind)`, not on
    prompt bytes, so that `make judge` works on a stranger's machine. Prompt-hash
    drift is measured and reported rather than hidden.
  - The L3 visual verifier's measured error rate on the occlusion scene.
- **Citations** (both verified against arXiv on 2026-08-29 — use exactly these):
  - Zhi, P., Zhang, Z., Han, M., Zhang, Z., Li, Z., Jiao, Z., et al. *Closed-Loop
    Open-Vocabulary Mobile Manipulation with GPT-4V* (COME-Robot), arXiv:2404.10220.
    The first closed-loop robotic system using a VLM for open-ended reasoning and
    adaptive replanning in the real world.
  - *FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models*,
    arXiv:2510.01642. Generates failure cases paired with executable recovery actions;
    reports up to +22.6% average improvement across three state-of-the-art VLA models
    on ManiSkill.
  One sentence:
  the field already knows open-loop VLM execution is unreliable; this project is the
  rigorously measured, reproducible engineering of the known fix. We are not claiming
  novel research, and saying so is a strength.
- **Hot take**, drawn from the measured data — the candidate, to be confirmed or
  overturned by the numbers: the cheapest verification layer catches the most
  failures, and the expensive one mostly catches the *dangerous* ones. Verification
  should be ordered by cost, and the honesty gap — not the success rate — is the
  metric that separates a demo from a system.

- [ ] **Step 5b: Freeze an exact lock file**

`requirements.txt` pins our direct dependencies but not their transitives (installing
`pytest==8.3.3` pulled in unpinned `iniconfig`, `packaging`, `pluggy`). Reproducibility
is 15% of the rubric and "it worked on my machine six months ago" is exactly what a
lock file is for:

```bash
VIRTUAL_ENV=.venv uv pip freeze > requirements.lock.txt
git add requirements.lock.txt && git commit -m "chore: freeze exact dependency lock"
```

Keep BOTH files and say why in `REPRODUCTION.md`: `requirements.txt` is the readable
statement of intent that `make setup` installs, and `requirements.lock.txt` is the
exact byte-for-byte environment the reported numbers were produced on. If a judge
gets a different result, diffing against the lock is the first thing to try.

- [ ] **Step 6: Write `REPRODUCTION.md` for a stranger on a clean machine**

Exact commands, pinned versions, the macOS pybullet `CFLAGS` fix and *why* it is
needed, expected output, and the measured runtime and cost for both modes: replay
(free, minutes, no API key) and live (measured dollars, measured minutes, needs
`GEMINI_API_KEY`). State the tested platform explicitly.

- [ ] **Step 7: Archive the coding-agent traces**

Save this session's transcript and the plan under `docs/agent-traces/`, with a short
`README.md` naming every agent used and what it did. This is a required disclosure.

- [ ] **Step 8: Fresh-clone dry run — do this before declaring done**

```bash
cd /tmp && rm -rf repro-check && git clone <repo> repro-check && cd repro-check
make setup && make test && make judge
```
Expected: `make judge` reproduces the reported headline numbers on a clean checkout
with **no API key set**. An unreproducible project is not marked down — it is
disqualified before scoring. Fix anything this surfaces before anything else.

- [ ] **Step 9: Final commit**

```bash
git add -A && git commit -m "docs: README, reproduction guide, changelog, agent traces"
```

---

## Self-review

**Spec coverage.** Every numbered step of the brief maps to a task: spike (done, ✅
passing) → primitives (Tasks 6–7) → baseline (Task 9) → harness + 10 scenes (Tasks 3,
12) → agent with L1/L2/L3, memory, escalation (Tasks 10–11) → evidence pass and the
removed experiment (Task 14) → trajectory pages and packaging (Tasks 13–14). The
firewall is Task 1 and is re-checked in Tasks 4, 5, and 9.

**Known gaps deliberately left to implementation time.** `harness/report.py` and
`harness/trajectory.py` are specified by required *contents* rather than by full source
(Task 13) — they are presentation code with no behavioural contract worth pinning in a
test, and writing 300 lines of HTML string-building into this plan would be noise. Every
module with a behavioural contract has complete code and a test.

**Type consistency.** `Feedback` fields are used identically in `primitives/api.py`,
`agent/verify.py`, `agent/react.py`, and `harness/trajectory.py`. `EpisodeResult` is
constructed only in `harness/episode.py` and read in `harness/metrics.py` and
`harness/report.py`. `VerificationConfig.label` is the single source of the condition
name used for both cache keys and report rows.

**One structural issue is flagged rather than hidden:** Task 9's `run_baseline` as first
drafted imports `robotsim.oracle` from inside `agent/`, which the firewall test forbids.
The fix is stated inline — move episode running and scoring into `harness/episode.py`
and leave `agent/` holding policy only. That is the correct architecture, and the
firewall test failing is the test doing its job.
