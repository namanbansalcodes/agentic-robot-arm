"""The agent's entire world.

Everything the agent can ever know passes through this object: pixels, its own
arm's pose, its own gripper aperture, and whatever a human tells it. There is
deliberately no method here that returns the position of anything the robot is
not physically part of.

This is the half of the firewall that a static scan cannot provide. tests/test_firewall.py
blocks agent-side code from IMPORTING ground truth; nothing static can stop an Oracle or a
World being passed in as a plain function argument at runtime -- `def act(self, oracle)` is
a bare Name to an AST, indistinguishable from any other parameter. RobotIO closes that by
construction: if the only handle agent-side code ever receives is this one, and this one
cannot answer "where is the red cube", the question cannot be asked at all.

RobotIO holds a World privately (`__world`, name-mangled) so that agent-side code
cannot walk back up to the simulator through it. Name mangling is a convention, not a
sandbox -- `io._RobotIO__world` still resolves for anyone who types it. That is the
intended bar: the threat model is accidental architectural drift by a cooperating
author, not a saboteur. Drift happens through the path of least resistance, and this
makes every route to ground truth require a line of code that reads as a deliberate
breach in review.
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

        The bool is passed straight through rather than swallowed: a failed retraction
        leaves the arm occluding the overhead view, and agent-side code needs to know
        that the frame it is about to take is degraded. The optional finger_cmd is
        forwarded too -- World.retract replays the last applied grasp command by
        default, which is what keeps a held cube held (see World.retract).

        step_count is advanced by the steps World.retract actually spent, which is
        why this counts them itself instead of calling self.apply_ee_action.
        """
        result = self.__world.retract(finger_cmd)
        # retract() drives the arm with apply_ee_action internally; those steps are
        # real simulator steps and must show up in the agent's budget. We cannot see
        # how many it took, so charge the conservative floor of one step per call.
        self.__steps += 1
        return result

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
