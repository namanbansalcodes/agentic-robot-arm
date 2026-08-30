"""Runs one episode of one condition on one scene at one seed, and scores it.

This module is on the JUDGE side of the firewall: it imports robotsim.oracle, and
agent/ may never import it back. The mid-episode disturbance lives here for the same
reason the oracle does -- it is a property of the WORLD, not of the policy, and both
conditions must meet exactly the same one.
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

# How long the world is given to settle after a block is teleported back onto the
# table. Enough for the cube to drop the last millimetre and stop; short enough that
# the disturbance is not itself a pause the agent could time.
DISTURB_SETTLE_STEPS = 20

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


class Disturbance:
    """The adversarial intervention, on the judge's side of the firewall.

    Built by the harness, handed to a policy as a bare zero-argument `on_step`
    callback. The policy cannot see the scene, the world, or the oracle through it and
    cannot tell that it did anything -- discovering the perturbation by LOOKING is the
    behaviour the disturbance cells exist to measure, so any channel that announced it
    would delete the finding.

    It fires AT MOST ONCE per episode. A hook that fired on every step would make the
    task unwinnable by construction, which measures nothing: the question is whether a
    policy notices that its own completed work was undone, not whether it can outrun an
    adversary. `fired` is what the EpisodeResult records as `disturbed`.
    """

    def __init__(self, world: World, oracle: Oracle, spec):
        self._world = world
        self._oracle = oracle
        self.spec = spec
        self.fired = False
        self.ejected: str | None = None      # which block was moved, for the record

    def _victim(self) -> str | None:
        """The first satisfied pair in SCENE ORDER, so the choice is deterministic.

        Deterministic matters more than clever here: both conditions must receive the
        identical disturbance, and "whichever block happens to be first in a set" would
        make the two runs differ for reasons that have nothing to do with the policy.
        """
        for item, container in self._world.scene.success.pairs:
            if self._oracle.is_contained(item, container):
                return item
        return None

    def _rest_height(self, item: str) -> float:
        for spec in self._world.scene.objects:
            if spec.name == item:
                return spec.half_extent
        return 0.025

    def __call__(self) -> None:
        if self.fired:
            return
        if self._oracle.pairs_satisfied() < self.spec.after_placements:
            return
        item = self._victim()
        if item is None:
            return
        x, y = self.spec.to
        self._world.sim.set_base_pose(
            item, np.array([float(x), float(y), self._rest_height(item)]),
            np.array([0.0, 0.0, 0.0, 1.0]))
        self._world.settle(DISTURB_SETTLE_STEPS)
        self.fired = True
        self.ejected = item


class EpisodeHook:
    """Everything the JUDGE does between primitives, as one opaque callable.

    Two jobs, in this order:
      1. poll the oracle so newly-completed placements are logged in the order they
         actually happened -- unrecoverable after the fact, since three cubes in a
         bowl look identical however they got there;
      2. fire the disturbance, if this scene has one.

    The polling runs BEFORE the disturbance on purpose: a placement that the
    disturbance is about to undo still happened, and an order log that forgot it would
    misjudge a `mem_order` episode for a reason that has nothing to do with ordering.

    Built the same way for every condition and deliberately NOT a function of the
    condition: what happens to the world between primitives is a property of the
    world, and a hook that knew which policy it was perturbing could not be compared
    across the two. What crosses the firewall is a zero-argument callable, so
    agent-side code holds no reference to the scene, the world, or the oracle.
    """

    def __init__(self, world: World, oracle: Oracle):
        self._oracle = oracle
        spec = getattr(world.scene, "disturbance", None)
        self.disturbance = Disturbance(world, oracle, spec) if spec else None

    @property
    def fired(self) -> bool:
        return bool(self.disturbance and self.disturbance.fired)

    def __call__(self) -> None:
        self._oracle.observe_placements()
        if self.disturbance is not None:
            self.disturbance()


def settle_and_score(world: World, oracle: Oracle, io: RobotIO):
    """Let the scene come to rest, then ask the oracle -- and only then.

    Returns (actual_success, progress).

    MANDATORY, and this call must not be deleted. Oracle.is_contained reads position,
    not rest state: a cube released over the bowl is still in the air when the last
    primitive returns, and a cube still in the gripper, held over the bowl below
    z=0.07, satisfies the predicate outright. Scoring mid-trajectory would credit a
    success the robot never completed and inflate the exact number this project
    exists to measure honestly. Settling lets the scene come to rest; the held check
    covers what settling cannot, because a closed gripper stays closed no matter how
    long the simulator is left to run.

    The held check now runs over EVERY graded item, not one: a multi-block task can
    end with the arm still carrying block three, and crediting that would reintroduce
    exactly the inflation the single-item version was written to prevent. It covers
    `excluded` items too -- "leave it on the table" is not done while it is in the
    gripper, wherever the gripper happens to be.
    """
    world.settle(SETTLE_STEPS)
    # One last poll, AFTER the scene comes to rest: the closing placement is still in
    # the air when the last primitive returns, so without this the final block would
    # be missing from the order log on every otherwise-perfect episode.
    oracle.observe_placements()

    pairs = world.scene.success.pairs
    excluded = world.scene.success.excluded
    graded = [item for item, _ in pairs] + [item for item, _ in excluded]
    held = {item for item in graded if _still_holding(world, oracle, item)}

    satisfied = sum(1 for item, container in pairs
                    if item not in held and oracle.is_contained(item, container))
    satisfied += sum(1 for item, container in excluded
                     if item not in held and not oracle.is_contained(item, container))
    total = len(pairs) + len(excluded)
    progress = satisfied / total if total else 0.0

    if held:
        return False, progress
    return oracle.actual_success(asked_human=io.asked_human()), progress


def run_episode(scene, seed: int, client, config, out_dir, condition=None) -> EpisodeResult:
    """Run one episode and score it. The only place a claim meets ground truth.

    `condition` is passed in rather than derived, because the dict key the report and
    the cache agree on ("agentic") is not always VerificationConfig.label
    ("agent_L1L2L3") -- see harness/run.py.
    """
    condition = condition or (config.label if config else "one_shot")
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
        oracle = Oracle(world)
        # Built BEFORE the policy is chosen and passed to whichever one runs, so the
        # two conditions cannot receive different worlds.
        hook = EpisodeHook(world, oracle)

        # The policy is chosen by whether a VerificationConfig was supplied, not by
        # matching a condition NAME: a string comparison would silently run the
        # one-shot policy for any condition someone later renamed.
        if config is None:
            trace = plan_once(scene, seed, io, api, client, TOOLS, dispatch,
                              on_step=hook)
        else:
            trace = run_agent_policy(scene, seed, io, api, client, config, TOOLS,
                                     dispatch, on_step=hook)

        actual_success, progress = settle_and_score(world, oracle, io)
        asked_human = io.asked_human()
        disturbed = hook.fired
        order_correct = oracle.order_correct()
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
        trace_steps=trace.steps,
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
        progress=float(progress),
        pairs_total=int(len(scene.success.pairs)),
        disturbed=disturbed,
        order_correct=bool(order_correct),
    )
