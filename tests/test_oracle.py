"""Ground truth, exercised directly. Teleports rather than trajectories: what is under
test here is the PREDICATE, and driving the arm to set up each case would test the arm.
"""
import dataclasses

import numpy as np
import pytest

from harness.scenes import SuccessSpec, load_scenes
from robotsim.oracle import Oracle
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}
UPRIGHT = np.array([0.0, 0.0, 0.0, 1.0])


# Four in-bowl slots, as a fraction of the bowl's interior. Cubes teleported to one
# point would land inside each other and get shoved out by the solver, so a helper
# that "solves" a multi-block scene has to spread them exactly as `place` does.
SLOTS = ((-1, -1), (1, 1), (1, -1), (-1, 1))
# Free table spots, well clear of every bowl, for blocks that must end up OUT.
TABLE_SPOTS = ((0.14, -0.02), (0.00, -0.02), (-0.14, -0.02))


def _drop_in(world, item, bowl, slot=0):
    cx, cy, r, h = world._bowl_centers[bowl]
    offset = (r - 0.008 - 0.025) * 0.7
    dx, dy = SLOTS[slot % len(SLOTS)]
    world.sim.set_base_pose(item, np.array([cx + dx * offset, cy + dy * offset, h * 0.6]),
                            UPRIGHT)


def _put_on_table(world, item, xy=(0.14, -0.02)):
    world.sim.set_base_pose(item, np.array([xy[0], xy[1], 0.025]), UPRIGHT)


def _solve(world, scene):
    """Teleport the scene into its goal state, one graded condition at a time.

    Excluded blocks are moved OUT first: on mem_recall the pre-placed block occupies
    the bowl at spawn, and filling the bowl around it would jam three cubes into a
    space sized for two.
    """
    for item, spot in zip((i for i, _ in scene.success.excluded), TABLE_SPOTS):
        _put_on_table(world, item, spot)
    used = {}
    for item, container in scene.success.pairs:
        _drop_in(world, item, container, used.get(container, 0))
        used[container] = used.get(container, 0) + 1
    world.settle(20)


# --- the predicate ------------------------------------------------------------

def test_untouched_scene_is_not_a_success():
    world = World(SCENES["h1_single"], seed=0)
    oracle = Oracle(world)
    assert oracle.actual_success(asked_human=False) is False
    world.close()


def test_cube_teleported_into_bowl_is_a_success():
    world = World(SCENES["h1_single"], seed=0)
    oracle = Oracle(world)
    _drop_in(world, "red_cube", "blue_bowl")
    world.settle(20)
    assert oracle.actual_success(asked_human=False) is True
    world.close()


def test_every_pair_must_hold_not_just_one():
    """The multi-block predicate in one assertion: two of three is a failure."""
    world = World(SCENES["h3_triple"], seed=0)
    oracle = Oracle(world)
    _drop_in(world, "red_cube", "blue_bowl", 0)
    _drop_in(world, "green_cube", "blue_bowl", 1)
    world.settle(20)
    assert oracle.pairs_satisfied() == 2 and oracle.total_pairs() == 3
    assert oracle.actual_success(asked_human=False) is False
    world.close()


def test_unknown_success_type_raises():
    """A scene whose predicate the oracle does not implement must stop the run, not
    quietly score zero for everyone and look like a hard task."""
    scene = dataclasses.replace(SCENES["h1_single"],
                                success=SuccessSpec(type="vibes",
                                                    pairs=(("red_cube", "blue_bowl"),)))
    world = World(scene, seed=0)
    with pytest.raises(ValueError, match="unknown success type"):
        Oracle(world).actual_success(asked_human=False)
    world.close()


# --- progress -----------------------------------------------------------------

@pytest.mark.parametrize("scene_id", sorted(SCENES))
def test_progress_is_zero_at_spawn_and_one_when_solved(scene_id):
    """Partial credit is the metric the horizon ladder is read off, so its two
    endpoints are pinned on EVERY scene -- including the two that start with blocks
    already in bowls, where a predicate that forgot the starting state would read 0.5
    before the robot had done anything at all."""
    scene = SCENES[scene_id]
    world = World(scene, seed=0)
    oracle = Oracle(world)
    assert oracle.progress() == 0.0, f"{scene_id}: not zero at spawn"
    _solve(world, scene)
    assert oracle.progress() == 1.0, f"{scene_id}: not one when solved"
    world.close()


def test_progress_is_a_fraction_of_the_graded_conditions():
    """One of h3_triple's three graded pairs satisfied is 1/3, not 0 -- which is the
    whole reason progress exists alongside task success."""
    world = World(SCENES["h3_triple"], seed=0)
    oracle = Oracle(world)
    _drop_in(world, "red_cube", "blue_bowl")
    world.settle(20)
    assert oracle.progress() == pytest.approx(1 / 3)
    world.close()


# --- ordering -----------------------------------------------------------------

def test_ordered_scene_accepts_the_required_sequence():
    scene = SCENES["mem_order"]
    world = World(scene, seed=0)
    oracle = Oracle(world)
    for slot, (item, container) in enumerate(scene.success.pairs):
        _drop_in(world, item, container, slot)
        world.settle(20)
        oracle.observe_placements()          # the harness polls after every primitive
    assert oracle.placement_order == ["red_cube", "green_cube", "yellow_cube"]
    assert oracle.order_correct() is True
    assert oracle.actual_success(asked_human=False) is True
    world.close()


def test_ordered_scene_rejects_the_right_set_in_the_wrong_order():
    """The distinction the scene exists for: every block is in the bowl, and it is
    still a failure -- but a DIFFERENT failure from not finishing, which is why
    order_correct is reported next to actual_success rather than folded into it."""
    scene = SCENES["mem_order"]
    world = World(scene, seed=0)
    oracle = Oracle(world)
    for slot, item in enumerate(("green_cube", "red_cube", "yellow_cube")):
        _drop_in(world, item, "blue_bowl", slot)
        world.settle(20)
        oracle.observe_placements()
    assert oracle.pairs_satisfied() == oracle.total_pairs()
    assert oracle.progress() == 1.0
    assert oracle.order_correct() is False
    assert oracle.actual_success(asked_human=False) is False
    world.close()


def test_order_is_vacuously_correct_where_no_order_is_required():
    world = World(SCENES["h3_triple"], seed=0)
    oracle = Oracle(world)
    _drop_in(world, "yellow_cube", "blue_bowl")
    world.settle(20)
    oracle.observe_placements()
    assert oracle.order_correct() is True
    world.close()


def test_placement_order_records_a_block_once_even_if_it_leaves_and_returns():
    """The log answers "in what order did these first arrive". A block ejected by the
    disturbance and put back is not a second arrival."""
    scene = SCENES["disturb_h3"]
    world = World(scene, seed=0)
    oracle = Oracle(world)
    _drop_in(world, "red_cube", "blue_bowl"); world.settle(20)
    oracle.observe_placements()
    _put_on_table(world, "red_cube"); world.settle(20)
    oracle.observe_placements()
    _drop_in(world, "red_cube", "blue_bowl"); world.settle(20)
    assert oracle.observe_placements() == ["red_cube"]
    world.close()


# --- recall -------------------------------------------------------------------

def test_recall_needs_the_pre_placed_block_taken_back_out():
    scene = SCENES["mem_recall"]
    world = World(scene, seed=0)
    oracle = Oracle(world)
    assert oracle.is_contained("red_cube", "blue_bowl"), \
        "precondition: mem_recall starts with red_cube already in the bowl"
    for slot, (item, container) in enumerate(scene.success.pairs, start=1):
        _drop_in(world, item, container, slot)
    world.settle(20)
    assert oracle.pairs_satisfied() == oracle.total_pairs()
    assert oracle.actual_success(asked_human=False) is False, \
        "the pre-placed block is still in the bowl"
    assert oracle.progress() == pytest.approx(2 / 3)
    _put_on_table(world, "red_cube")
    world.settle(20)
    assert oracle.actual_success(asked_human=False) is True
    assert oracle.progress() == 1.0
    world.close()
