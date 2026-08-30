"""Feasibility spike. Proves three things before any real code gets written:

1. panda-gym / PyBullet runs headless on CPU (DIRECT mode, TinyRenderer).
2. A hard-coded scripted grasp actually picks up a block.
3. We can render an RGB camera image of the table.

Run: make spike     (or  .venv/bin/python spike/spike.py)
"""
import os
import sys
import time

import numpy as np
from PIL import Image

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda

OUT = os.path.join(os.path.dirname(__file__), "out")
BLOCK_HALF = 0.025          # 5cm block -- big and easy on purpose
BLOCK_START = np.array([0.05, 0.0, BLOCK_HALF])
LIFT_Z = 0.20


def servo_to(robot, sim, target_xyz, fingers, steps=120, tol=0.005):
    """Proportional servo in EE space. Our code computes WHERE; the VLM never does."""
    for _ in range(steps):
        ee = robot.get_ee_position()
        delta = np.asarray(target_xyz) - ee
        if np.linalg.norm(delta) < tol and steps > 20:
            break
        action = np.concatenate([np.clip(delta * 8.0, -1.0, 1.0), [fingers]])
        robot.set_action(action)
        sim.step()
    return robot.get_ee_position()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    sim = PyBullet(render_mode="rgb_array", n_substeps=20, renderer="Tiny")
    with sim.no_rendering():
        sim.create_plane(z_offset=-0.4)
        sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.05)
        sim.create_box(
            body_name="block",
            half_extents=np.array([BLOCK_HALF] * 3),
            mass=0.08,
            position=BLOCK_START,
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
            lateral_friction=2.0,
            spinning_friction=0.05,
        )
        robot = Panda(sim, block_gripper=False,
                      base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    robot.reset()
    sim.set_base_pose("block", BLOCK_START, np.array([0.0, 0.0, 0.0, 1.0]))
    for _ in range(20):
        sim.step()
    print(f"[1/3] headless sim up. ee={robot.get_ee_position().round(3)}")

    img = sim.render(width=480, height=360, target_position=np.array([0.0, 0.0, 0.05]),
                     distance=0.9, yaw=45.0, pitch=-35.0, roll=0.0)
    assert img is not None and img.shape[0] == 360, f"bad render: {img}"
    Image.fromarray(img[:, :, :3]).save(os.path.join(OUT, "camera.png"))
    print(f"[2/3] rendered RGB {img.shape} -> spike/out/camera.png")

    block = sim.get_base_position("block")      # spike only; the agent NEVER sees this
    servo_to(robot, sim, np.array([block[0], block[1], 0.15]), fingers=1.0, steps=150)
    servo_to(robot, sim, np.array([block[0], block[1], BLOCK_HALF + 0.005]), fingers=1.0, steps=120)
    for _ in range(40):
        robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
        sim.step()
    servo_to(robot, sim, np.array([block[0], block[1], LIFT_Z]), fingers=-1.0, steps=150)

    final_z = sim.get_base_position("block")[2]
    width = robot.get_fingers_width()
    lifted = final_z > BLOCK_HALF + 0.05
    print(f"[3/3] grasp: block_z={final_z:.3f} finger_width={width:.4f} lifted={lifted}")

    img2 = sim.render(width=480, height=360, target_position=np.array([0.0, 0.0, 0.1]),
                      distance=0.9, yaw=45.0, pitch=-35.0, roll=0.0)
    Image.fromarray(img2[:, :, :3]).save(os.path.join(OUT, "after_grasp.png"))

    print(f"\nwall time: {time.time()-t0:.1f}s")
    if not lifted:
        print("SPIKE FAIL: scripted grasp did not lift the block")
        sys.exit(1)
    print("SPIKE PASS: headless + grasp + camera all work")


if __name__ == "__main__":
    main()
