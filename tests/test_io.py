import numpy as np
import pytest

from robotsim.io import RobotIO
from robotsim.world import World
from tests import SCENES, scene_with_human_answer

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
    world = World(SCENES["h1_single"], seed=0)
    io = RobotIO(world)
    for forbidden in ("get_base_position", "sim", "world", "scene", "oracle"):
        assert not hasattr(io, forbidden), f"RobotIO leaks {forbidden}"
    world.close()


def test_proprioception_is_available():
    world = World(SCENES["h1_single"], seed=0)
    io = RobotIO(world)
    assert io.ee_position().shape == (3,)
    assert 0.0 <= io.fingers_width() <= 0.09
    assert len(io.joint_positions()) == 7
    assert io.render("overhead").shape == (480, 480, 3)
    world.close()


def test_ask_human_returns_the_scripted_answer_and_is_recorded():
    world = World(scene_with_human_answer(SCENES["h1_single"],
                                          "Use the blue bowl."), seed=0)
    io = RobotIO(world)
    assert io.asked_human() is False
    answer = io.ask_human("Which bowl should I use?")
    assert answer == "Use the blue bowl."
    assert io.asked_human() is True
    world.close()


def test_ask_human_on_an_unambiguous_scene_says_so():
    world = World(SCENES["h1_single"], seed=0)
    io = RobotIO(world)
    answer = io.ask_human("Which bowl?")
    assert "no additional" in answer.lower() or "instruction" in answer.lower()
    world.close()
