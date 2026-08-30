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
