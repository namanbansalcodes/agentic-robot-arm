"""The robot's hands. The only way a VLM ever touches the arm.

Contract, enforced by design:
  * the caller supplies OBJECT IDS. Never coordinates.
  * this module computes every pose, from pixels, via camera unprojection.
  * every primitive returns a Feedback -- including the failures.

The first clause is the whole point. A VLM that can emit `grasp(0.02, -0.05)` is being
graded on its ability to guess millimetres out of a 480x480 photo, which is a
measurement of the renderer, not of the agent. Every public method here takes an id
produced by `primitives.perception.detect` -- a handle, not a pose -- and this module
turns it into a target with `OVERHEAD.unproject`. `_grasp_at` is the only entry point
that accepts a raw point; it is private, it is called from `grasp()` and the tests, and
nothing in the tool schema handed to the model ever names it.

The second clause is why the failures are as detailed as the successes. Feedback is the
raw material the verification layers run on: an error string is L1, the gripper aperture
is L2, the saved overhead frame is L3. A primitive that failed silently, or that returned
None on the error path, would starve all three.
"""
from __future__ import annotations

import pathlib

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

# --- where inside a target to let go ---------------------------------------------
# MEASURED FAILURE that forced these into existence: `place` used to aim at the
# target's detected centroid, always. With one block and one bowl that is correct.
# With two it is fatal -- the second cube landed dead on top of the first at z=0.0910
# (measured, h2_pair) and the containment cutoff is height+0.02 = 0.0700, so a
# perfectly executed two-block episode scored 0.50 forever. Every multi-block scene in
# this eval was unreachable until this was fixed, so this is not a tuning knob; it is
# the primitive learning to do the task the scenes now ask for.
#
# Both rules are computed from PIXELS ONLY -- detected bounding boxes and centroids
# from the overhead frame -- so nothing here consults simulator state and the firewall
# is untouched.
#
# SLOT OFFSET: how far from the bowl's bbox centre, per axis, the
# four drop slots sit. A flat fraction does NOT work, and that was measured: at 0.20 of
# the bbox the offset is ~0.024 m in a small (r=0.060) bowl, and the cube landed perched
# on the rim at z=0.0750 -- the exact rim-perch the TUNING WARNING in scenes.yaml
# describes, which the containment predicate correctly rejects at 0.0700.
#
# So the offset is derived from what the picture actually shows. A bowl of outer
# half-width r has interior half-width r-0.008 and must swallow a 0.025 half-extent
# cube, so the largest safe per-axis offset is (r-0.008)-0.025. Expressed in pixels,
# using a detected CUBE as the ruler (a cube is 0.050 m across, so its bbox gives
# pixels-per-metre):
#     offset_px = (bowl_w/2 - WALL_IN_CUBES*cube_w - cube_w/2) - DROP_SLACK_IN_CUBES*cube_w
# WALL_IN_CUBES = 0.008/0.050 = 0.16 is the wall thickness in cube widths.
# DROP_SLACK_IN_CUBES is a SUBTRACTED allowance, not a scale factor, because what it
# covers does not scale with the bowl: a cube released above the rim drifts as it falls
# and lands about a centimetre off its target (measured: (+0.002, -0.009) m on a
# match3 placement), and the bowl rim also reads ~3% larger per metre than a cube does
# because it sits 0.05 m closer to the camera. 0.20 cube widths = 0.010 m.
#
# What that arithmetic then says, and it is a RESULT rather than a setting: a big
# (r=0.075) bowl gets a 0.032 m slot offset, so its four slots are 0.064 m apart and it
# holds four cubes; a small (r=0.060) bowl gets 0.017 m, so its slots are only 0.034 m
# apart and it can hold exactly one cube reliably. An r=0.060 bowl has a 0.104 m
# interior and two cubes need 0.100 m of it -- 4 mm of slack against 10 mm of drift.
#
# WITH NO CUBE VISIBLE the offset is zero: aim at the middle. This is not a degenerate
# case, it is a common one, and it cost a scene to find. A cube sitting in a bowl of
# its OWN colour merges into the bowl's colour blob and stops being a separate
# detection -- so on match3's third placement the frame contained no cube at all, the
# offset fell back to a flat fraction of the bowl, and the cube landed perched on the
# rim at z=0.0750 against a 0.0700 cutoff. Zero is right for both halves of that
# situation: there is no ruler to measure a safe offset with, and nothing visible
# inside the bowl to need avoiding.
WALL_IN_CUBES = 0.16
DROP_SLACK_IN_CUBES = 0.20
# BESIDE_STEPS: placing onto a CUBE means beside it on the table, not on top of it.
# A stack is not a placement anyone asked for, and beside-ness is the only way this
# primitive set can express "put this down out of the way" -- which mem_swap and
# mem_recall both require and no other primitive provides.
#
# Two radii and eight directions, not one spot, because MEASURED: parking a cube 1.4
# target-widths away from a block that was sitting in a bowl left it 0.107 m from the
# bowl centre, which is close enough that the two merged into ONE colour blob in the
# overhead frame (bbox 137 px where the bowl alone is 102) -- the parked cube stopped
# being detectable and could never be picked up again. A bowl is 0.05 m tall and
# projects a skirt from a camera 0.85 m up, so "clear in world coordinates" is not the
# same as "clear in the picture". Offering the chooser a farther ring lets it find
# genuinely open table.
BESIDE_STEPS = (1.4, 2.2)
BESIDE_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1),
                     (0.7071, 0.7071), (0.7071, -0.7071),
                     (-0.7071, 0.7071), (-0.7071, -0.7071))

OPEN = 1.0
CLOSE = -1.0
# Aperture below this == gripped air. Measured, not guessed: closed-on-air reads ~0.000,
# a held cube ~0.044, open-and-empty ~0.080. 0.012 sits in the middle of a 44 mm gap, so
# the L2 signal is a cliff rather than a threshold to be tuned. Widening it to rescue a
# marginal grasp would blunt exactly the signal L2 exists to measure.
EMPTY_GRIP_THRESHOLD = 0.012
SERVO_GAIN = 8.0
SERVO_TOL = 0.006


class PrimitiveAPI:
    def __init__(self, io: RobotIO, image_dir=None, episode_id: str = "ep"):
        self.io = io
        self.episode_id = episode_id
        self.image_dir = pathlib.Path(image_dir) if image_dir else pathlib.Path("results/images")
        self.image_dir.mkdir(parents=True, exist_ok=True)
        # The id -> Detection table from the most recent frame. This is the ONLY
        # vocabulary the caller may name, and it is empty until look() runs: naming a
        # thing requires having seen it in this episode's most recent photo.
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
        """Pixel centroid -> world point. OUR math, from OUR camera matrices.

        The z plane is an ASSUMPTION supplied here, not a lookup: cube centres sit a
        half-extent above the table, bowl rims at rim height. When the assumption is
        wrong -- a stacked cube, a bowl half hidden behind a wall -- the target is
        wrong and the grasp misses honestly, which is the failure the agent must catch.
        """
        z = BOWL_RIM_Z if det.kind == "bowl" else TABLE_Z + CUBE_HALF
        return OVERHEAD.unproject(det.centroid_px[0], det.centroid_px[1], z_plane=z)

    def _cube_centroids(self) -> list:
        """Every currently-detected cube's pixel centroid. Perception, not truth."""
        return [d.centroid_px for d in self._detections.values() if d.kind == "cube"]

    def _cube_width_px(self) -> float:
        """The ruler: how many pixels a 0.050 m cube covers in THIS frame.

        Median over every detected cube, so one clipped or partly occluded blob cannot
        set the scale on its own. Returns 0.0 when nothing is visible to measure.
        """
        widths = sorted(max(d.bbox_px[2], d.bbox_px[3])
                        for d in self._detections.values() if d.kind == "cube")
        return float(widths[len(widths) // 2]) if widths else 0.0

    def _slot_offset_px(self, bowl_span_px: float) -> float:
        """Largest per-axis offset that still leaves the cube clear of the bowl wall."""
        cube_px = self._cube_width_px()
        if not cube_px:
            return 0.0
        return max(0.0, bowl_span_px / 2.0
                   - (WALL_IN_CUBES + 0.5 + DROP_SLACK_IN_CUBES) * cube_px)

    def _clearest(self, candidates: list, occupied: list):
        """The reachable candidate farthest from the nearest occupied pixel.

        Candidates are (pixel, world_point) pairs and unreachable ones are dropped
        first: a release point the arm cannot get to is not a candidate, and choosing
        one and only then discovering it was out of bounds is how `place` used to fail
        with "unreachable" while a perfectly good spot sat one option away. Ties break
        on candidate ORDER, which is fixed, so the choice is deterministic -- both
        conditions must get the same drop point for the same picture, or the comparison
        is measuring the tie-break.
        """
        reachable = [c for c in candidates if self._in_workspace(c[1])]
        if not reachable:
            return None
        if not occupied:
            return reachable[0]
        return max(reachable,
                   key=lambda c: min((c[0][0] - o[0]) ** 2 + (c[0][1] - o[1]) ** 2
                                     for o in occupied))

    def _release_point(self, det: Detection):
        """Where to let go, as a world point. None if nothing reachable is available.

        A bowl gets one of four slots inset from its bounding box, chosen to be
        farthest from whatever is already inside it; slots are anchored on the BBOX
        centre rather than the mask centroid because a cube sitting in the bowl punches
        a hole in the bowl's colour mask and drags the centroid around, while the box
        stays put. A cube gets a spot BESIDE it, away from every other detected object
        -- bowls included, or "put this down out of the way" would cheerfully drop the
        block onto a bowl rim.
        """
        x, y, w, h = det.bbox_px
        cx, cy = x + w / 2.0, y + h / 2.0
        z = BOWL_RIM_Z if det.kind == "bowl" else TABLE_Z + CUBE_HALF

        if det.kind == "bowl":
            dx, dy = self._slot_offset_px(w), self._slot_offset_px(h)
            pixels = [(cx - dx, cy - dy), (cx + dx, cy + dy),
                      (cx + dx, cy - dy), (cx - dx, cy + dy)]
            occupied = [c for c in self._cube_centroids()
                        if x <= c[0] <= x + w and y <= c[1] <= y + h]
        else:
            span = max(w, h)
            pixels = [(cx + ux * span * k, cy + uy * span * k)
                      for k in BESIDE_STEPS for ux, uy in BESIDE_DIRECTIONS]
            occupied = [d.centroid_px for d in self._detections.values()
                        if d.id != det.id] + [det.centroid_px]

        candidates = [(px, OVERHEAD.unproject(px[0], px[1], z_plane=z))
                      for px in pixels]
        chosen = self._clearest(candidates, occupied)
        return None if chosen is None else chosen[1]

    def _in_workspace(self, p) -> bool:
        b = self.io.workspace_bounds()
        return (b["x"][0] <= p[0] <= b["x"][1]) and (b["y"][0] <= p[1] <= b["y"][1])

    def _servo(self, target, finger_cmd: float, max_steps: int = 140) -> float:
        """Proportional servo in EE space. Returns residual distance to target.

        The residual is returned rather than asserted on: a servo that ran out of steps
        short of its target is a fact the agent needs, and callers turn it into an
        explicit servo_timeout error instead of proceeding as if the move landed.
        """
        target = np.asarray(target, dtype=np.float64)
        for _ in range(max_steps):
            delta = target - self.io.ee_position()
            if np.linalg.norm(delta) < SERVO_TOL:
                break
            self.io.apply_ee_action(delta * SERVO_GAIN, finger_cmd)
        return float(np.linalg.norm(target - self.io.ee_position()))

    def _feedback(self, primitive, args, status, error=None, note=None) -> Feedback:
        """Build the Feedback, refreshing perception first.

        Every primitive ends here, success or failure, so the detection table the next
        call may name is always the one from the frame the agent was just shown.
        """
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
        # Retract FIRST, or the arm sits under the overhead camera and occludes a third
        # of the table -- verified: a cube went completely missing. Note the bare call:
        # retract() replays the last applied finger command, and passing a constant here
        # would ratchet a closed gripper open and drop a held cube ~11% of the time.
        # look() runs between grasp and place, so that bug would corrupt the flow.
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
                                  error=f"servo_timeout: end effector stopped "
                                        f"{residual * 100:.1f}cm short of the target")
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
                                  error=f"unknown_object: no detection named '{object_id}'. "
                                        f"Call look() first.")
        if det.kind == "bowl":
            return self._feedback("grasp", {"object_id": object_id}, "error",
                                  error=f"bad_target: '{object_id}' is a bowl, not a graspable block")
        p = self._world_xy(det)
        if not self._in_workspace(p):
            # The bounds go in the message on purpose: "unreachable" alone invites a
            # retry, and a retry cannot help. Naming the limits lets the agent conclude
            # the task is impossible and say so, which is what this scene grades.
            return self._feedback("grasp", {"object_id": object_id}, "error",
                                  error=f"unreachable: '{object_id}' lies outside the arm workspace "
                                        f"(x,y limits {self.io.workspace_bounds()})")
        fb = self._grasp_at(p)
        # _grasp_at reports itself as a raw-point grasp; relabel with what the caller
        # actually asked for, so the transcript records ids rather than the internal call.
        fb.primitive, fb.args = "grasp", {"object_id": object_id}
        return fb

    def place(self, target_id: str) -> Feedback:
        det = self._detections.get(target_id)
        if det is None:
            return self._feedback("place", {"target_id": target_id}, "error",
                                  error=f"unknown_object: no detection named '{target_id}'")
        # The release point is a FREE spot in/next to the target, not the target's
        # centroid -- see _release_point, which also drops any candidate the arm
        # cannot reach, so "unreachable" here means EVERY spot was out of bounds.
        p = self._release_point(det)
        if p is None:
            return self._feedback("place", {"target_id": target_id}, "error",
                                  error=f"unreachable: no free space next to "
                                        f"'{target_id}' lies inside the arm workspace")
        self._servo([p[0], p[1], CARRY_Z], finger_cmd=CLOSE)
        self._servo([p[0], p[1], RELEASE_Z], finger_cmd=CLOSE, max_steps=90)
        for _ in range(30):
            self.io.apply_ee_action([0.0, 0.0, 0.0], OPEN)
        self._servo([p[0], p[1], CARRY_Z], finger_cmd=OPEN, max_steps=90)
        # Settle before the frame is taken: a cube released at rim height is still in
        # the air when the servo returns, and a photo of it mid-drop would let both the
        # agent and the visual verifier call a bounce-out a success.
        self.io.settle(25)
        return self._feedback("place", {"target_id": target_id}, "ok")

    def ask_human(self, question: str) -> Feedback:
        answer = self.io.ask_human(question)
        return self._feedback("ask_human", {"question": question}, "ok",
                              note=f"human replied: {answer}")

    def report_done(self, success: bool, reason: str) -> Feedback:
        """The claim. Compared against the oracle to compute the honesty gap."""
        return self._feedback("report_done", {"success": success, "reason": reason}, "ok",
                              note=f"agent claims success={success}: {reason}")
