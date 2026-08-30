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
# A bowl's floor plate is a box of half-thickness 0.008 centred at z=0.008, so its top
# surface is at 0.016. A cube spawned inside a bowl rests on that, not on the table.
BOWL_FLOOR_HALF_THICKNESS = 0.008
BOWL_FLOOR_TOP_Z = TABLE_Z + 2 * BOWL_FLOOR_HALF_THICKNESS
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
        self.body_names: list[str] = []          # logical scene vocabulary. NOTE: a bowl
        # is five sim bodies -- {name}__base and {name}__wall0..3 -- so a bowl's entry
        # here is NOT a valid argument to sim.get_base_position().
        self._bowl_centers: dict = {}

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
        # The gripper command most recently applied. retract() replays it so that
        # moving the arm never silently changes what the gripper is doing.
        self._last_finger_cmd = 0.0
        self.settle(30)

    # --- construction ---------------------------------------------------
    def _spawn(self, spec: ObjectSpec) -> None:
        rgba = np.array(COLORS[spec.color])
        # Jitter is seeded, small, and applied to every seed alike -- it is what
        # makes 3 seeds per scene 3 genuinely different episodes rather than 3 copies.
        jitter = self.rng.uniform(-0.012, 0.012, size=2)
        x, y = np.array(spec.position) + jitter

        if spec.kind == "cube":
            z = TABLE_Z + spec.half_extent
            if spec.in_bowl:
                # Spawn INSIDE a bowl that has already been built. The cube takes the
                # bowl's own (jittered) centre rather than a second independent draw,
                # and sits on the bowl's floor plate rather than on the table: two
                # jitter draws could otherwise put a cube far enough off centre to
                # start outside the containment predicate, which would make the
                # STARTING state of the memory scenes a per-seed coin flip.
                if spec.in_bowl not in self._bowl_centers:
                    raise ValueError(
                        f"{spec.name}: in_bowl='{spec.in_bowl}' is not a bowl declared "
                        "earlier in this scene's object list")
                bx, by, _, _ = self._bowl_centers[spec.in_bowl]
                x, y = bx, by
                z = BOWL_FLOOR_TOP_Z + spec.half_extent
            self.sim.create_box(
                body_name=spec.name,
                half_extents=np.array([spec.half_extent] * 3),
                mass=spec.mass,
                position=np.array([x, y, z]),
                rgba_color=rgba,
                lateral_friction=spec.lateral_friction,
                spinning_friction=0.05,
            )
            self.body_names.append(spec.name)

        elif spec.kind == "bowl":
            r, h, t = spec.radius, spec.height, BOWL_FLOOR_HALF_THICKNESS
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
        self._last_finger_cmd = float(finger_cmd)
        action = np.concatenate([np.clip(np.asarray(delta_xyz), -1.0, 1.0), [finger_cmd]])
        self.robot.set_action(action)
        self.sim.step()

    def ee_position(self) -> np.ndarray:
        return self.robot.get_ee_position()

    def fingers_width(self) -> float:
        return float(self.robot.get_fingers_width())

    def joint_positions(self) -> list:
        return [float(self.robot.get_joint_angle(i)) for i in range(7)]

    def retract(self, finger_cmd: float | None = None) -> bool:
        """Park the arm clear of the overhead camera. Called before every look().

        Returns True if the arm actually reached the home pose.

        The finger command REPLAYS whatever was last applied, because panda-gym does
        NOT treat 0.0 as "hold". Its action is relative to the width being measured
        right now (panda.py: target = get_fingers_width() + action[-1] * 0.2), and
        while a cube is squeezed the measured width sits ~5mm under the free width
        from contact penetration. Targeting that measured value releases the squeeze,
        the fingers spring wider, and the next step targets the new wider value -- it
        ratchets open at ~+4.9mm per retraction and drops the cube about 11% of the
        time (measured 24/27 held with 0.0, versus 27/27 replaying the grasp command).
        Since look() calls retract(), that bug would have silently corrupted the
        grasp -> look -> place flow, worst of all on the hard-grasp scenes.
        """
        cmd = self._last_finger_cmd if finger_cmd is None else finger_cmd
        target = np.array(HOME_RETRACT)
        for _ in range(200):
            delta = target - self.ee_position()
            if np.linalg.norm(delta) < 0.02:
                return True
            self.apply_ee_action(delta * 8.0, cmd)
        # Exhausting the loop leaves the arm somewhere over the table, occluding the
        # very view retracting exists to clear. Report it rather than returning a
        # silently degraded frame.
        return bool(np.linalg.norm(target - self.ee_position()) < 0.02)

    def render(self, camera_name: str = "overhead") -> np.ndarray:
        cameras = {"overhead": OVERHEAD, "oblique": OBLIQUE}
        if camera_name not in cameras:
            raise ValueError(f"unknown camera {camera_name!r}; expected one of {sorted(cameras)}")
        return cameras[camera_name].render(self.sim.physics_client._client)

    def close(self) -> None:
        self.sim.close()
