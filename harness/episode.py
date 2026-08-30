"""Runs one episode of one condition on one scene at one seed, and scores it.

This module is on the JUDGE side of the firewall: it imports robotsim.oracle, and
agent/ may never import it back.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

from agent.baseline import TOOLS, dispatch, plan_once
from agent.react import run_agent_policy
from harness.metrics import EpisodeResult
from primitives.api import PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.oracle import Oracle
from robotsim.world import World

# How long the scene gets to come to rest before ground truth is read. 60 steps is
# comfortably past the settling time of a cube dropped from the release height
# (measured: motion stops well inside 40), and costs a few milliseconds.
SETTLE_STEPS = 60

# "Still in the gripper" as two measurements the harness can make without asking the
# simulator anything the agent could not also be told. A free gripper reads 0.080 m
# and a held cube 0.044; a placed cube sits ~0.155 m from the parked end effector and
# a dangling one ~0.018 m. Both thresholds sit in the middle of those gaps, so this is
# a cliff, not a tuned margin.
HELD_APERTURE_M = 0.06
HELD_DISTANCE_M = 0.08


def _still_holding(world: World, oracle: Oracle, item: str) -> bool:
    """Is the graded object still clamped in the gripper?

    Oracle.is_contained reads a POSITION. A cube dangling in a closed gripper inside
    the bowl's footprint below the height cutoff satisfies it perfectly -- measured:
    the predicate returns True while the robot is still holding the block. Settling
    does not fix this: PyBullet holds the last finger target, so the grip survives any
    number of quiet steps (measured too, and it is why settling alone is not enough).

    A task is not done while the robot is still carrying the thing. If you took the
    arm away the cube would go with it, so crediting this inflates the exact number
    this project exists to measure honestly.
    """
    aperture = world.fingers_width()
    if aperture >= HELD_APERTURE_M:
        return False                      # gripper is open; it is holding nothing
    distance = float(np.linalg.norm(
        np.asarray(oracle.position_of(item), dtype=float)
        - np.asarray(world.ee_position(), dtype=float)))
    return distance < HELD_DISTANCE_M


def settle_and_score(world: World, oracle: Oracle, io: RobotIO) -> bool:
    """Let the scene come to rest, then ask the oracle -- and only then.

    MANDATORY, and this call must not be deleted. Oracle.is_contained reads position,
    not rest state: a cube released over the bowl is still in the air when the last
    primitive returns, and a cube still in the gripper, held over the bowl below
    z=0.07, satisfies the predicate outright. Scoring mid-trajectory would credit a
    success the robot never completed and inflate the exact number this project
    exists to measure honestly. Settling lets the scene come to rest; the held check
    covers what settling cannot, because a closed gripper stays closed no matter how
    long the simulator is left to run.
    """
    world.settle(SETTLE_STEPS)
    if _still_holding(world, oracle, world.scene.success.item):
        return False
    return oracle.actual_success(asked_human=io.asked_human())


def run_episode(scene, seed: int, client, config, out_dir, condition=None) -> EpisodeResult:
    """Run one episode and score it. The only place a claim meets ground truth.

    `condition` is passed in rather than derived, because the dict key the report and
    the cache agree on ("agent") is not always VerificationConfig.label
    ("agent_L1L2L3") -- see harness/run.py.
    """
    condition = condition or (config.label if config else "baseline")
    out_dir = pathlib.Path(out_dir)
    episode_id = f"{condition}_{scene.id}_s{seed}"
    started = time.monotonic()

    # SceneSpec is a frozen dataclass and stays that way: the seed travels as an
    # argument to every policy that needs it, never as an attribute smuggled onto the
    # scene (that bug silently collapsed five seeds onto one cache key once already).
    world = World(scene, seed)
    try:
        io = RobotIO(world)
        api = PrimitiveAPI(io, image_dir=out_dir / "images", episode_id=episode_id)

        if condition == "baseline":
            trace = plan_once(scene, seed, io, api, client, TOOLS, dispatch)
        else:
            trace = run_agent_policy(scene, seed, io, api, client, config, TOOLS,
                                     dispatch)

        actual_success = settle_and_score(world, Oracle(world), io)
        asked_human = io.asked_human()
    finally:
        # Unconditional. A crashing episode that leaked its PyBullet client would
        # take the other 249 down with it long before the run finished.
        world.close()

    return EpisodeResult(
        condition=condition,
        scene_id=scene.id,
        seed=seed,
        failure_mode=scene.failure_mode,
        instruction=scene.instruction,
        claimed_success=bool(trace.claimed_success),
        actual_success=bool(actual_success),
        asked_human=bool(asked_human),
        recoveries=int(trace.recoveries),
        steps=len(trace.steps),
        vlm_calls=int(getattr(client, "calls", 0)),
        input_tokens=int(getattr(client, "input_tokens", 0)),
        output_tokens=int(getattr(client, "output_tokens", 0)),
        cost_usd=float(client.cost_usd()),
        drift=int(getattr(client, "drift_count", 0)),
        episode_id=episode_id,
        claim_reason=trace.claim_reason,
        stop_reason=trace.stop_reason,
        wall_seconds=time.monotonic() - started,
        l3_calls=int(trace.l3_calls),
    )
