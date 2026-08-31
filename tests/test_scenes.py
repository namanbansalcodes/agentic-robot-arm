"""scenes.yaml IS the experiment, so it is under test like code.

A typo here does not crash anything: it produces a scene that runs, scores, and
reports a number about the wrong thing. These tests exist to make that impossible.
"""
from harness.scenes import load_scenes
from robotsim.world import WORKSPACE

# One cell per scene. Horizon length and disturbance are the two independent
# variables; the three memory modes reproduce RoboVoLo's Order / Swap / Recall.
REQUIRED_FAILURE_MODES = {
    "horizon_1", "horizon_2", "horizon_3", "matching_3",
    "memory_order", "memory_swap", "memory_recall", "disturbance",
}
VALID_SUCCESS_TYPES = {"pairs", "ordered_pairs", "recall"}


def test_loads_the_declared_scene_set():
    scenes = load_scenes()
    assert len(scenes) == 9
    assert len({s.id for s in scenes}) == len(scenes), "scene ids must be unique"


def test_every_failure_mode_is_covered():
    modes = {s.failure_mode for s in load_scenes()}
    assert modes == REQUIRED_FAILURE_MODES


def test_defaults_are_applied():
    scene = {s.id: s for s in load_scenes()}["h1_single"]
    assert scene.max_steps == 24
    assert scene.max_retries_per_subtask == 3
    cube = [o for o in scene.objects if o.kind == "cube"][0]
    assert cube.half_extent == 0.025
    assert cube.lateral_friction == 2.0
    assert cube.radius == 0.075          # the default a bowl inherits


def test_per_object_bowl_radius_overrides_the_default():
    """The three-bowl scenes only fit at r=0.060; at the 0.075 default they overlap."""
    scene = {s.id: s for s in load_scenes()}["match3"]
    bowls = [o for o in scene.objects if o.kind == "bowl"]
    assert len(bowls) == 3
    assert all(o.radius == 0.060 for o in bowls)


def test_horizon_ladder_is_nested():
    """h1 subset h2 subset h3 subset h4, with the bowl in the same place throughout.

    If the ladder were not nested, a difference between two rungs could be a
    difference of layout rather than of horizon length, and the whole axis would stop
    meaning what the report says it means.
    """
    scenes = {s.id: s for s in load_scenes()}
    ladder = [scenes[i] for i in ("h1_single", "h2_pair", "h3_triple")]
    previous = set()
    for scene, expected in zip(ladder, (1, 2, 3, 4)):
        blocks = {(o.name, o.position) for o in scene.objects if o.kind == "cube"}
        assert len(blocks) == expected == len(scene.success.pairs), scene.id
        assert previous <= blocks, f"{scene.id} is not a superset of the rung below it"
        previous = blocks
        bowl = [o for o in scene.objects if o.kind == "bowl"]
        assert len(bowl) == 1 and bowl[0].position == (-0.02, 0.16), scene.id


def test_success_references_resolve_to_real_objects_of_the_right_kind():
    """A typo in a success pair would pass every other test in this file and silently
    break scoring -- the oracle would raise, or worse, score the wrong body."""
    for scene in load_scenes():
        kinds = {o.name: o.kind for o in scene.objects}
        pairs = list(scene.success.pairs) + list(scene.success.excluded)
        assert pairs, f"{scene.id}: no graded pairs"
        for item, container in pairs:
            assert kinds.get(item) == "cube", f"{scene.id}: bad item {item!r}"
            assert kinds.get(container) == "bowl", f"{scene.id}: bad container {container!r}"


def test_every_scene_declares_a_known_failure_mode_and_success_type():
    for scene in load_scenes():
        assert scene.failure_mode in REQUIRED_FAILURE_MODES, scene.id
        assert scene.success.type in VALID_SUCCESS_TYPES, scene.id


def test_only_the_recall_scene_carries_recall_fields():
    """`initially_in_bowl` and `excluded` are the recall predicate's two halves and
    must travel together -- one without the other scores a different task."""
    for scene in load_scenes():
        is_recall = scene.success.type == "recall"
        assert bool(scene.initially_in_bowl) == is_recall, scene.id
        assert bool(scene.success.excluded) == is_recall, scene.id
        if is_recall:
            assert [scene.initially_in_bowl] == [i for i, _ in scene.success.excluded]
            pre_placed = [o for o in scene.objects if o.name == scene.initially_in_bowl]
            assert pre_placed and pre_placed[0].in_bowl, \
                f"{scene.id}: {scene.initially_in_bowl} must actually spawn in a bowl"


def test_in_bowl_always_names_a_bowl_declared_earlier():
    """World builds objects in list order and resolves in_bowl against bowls it has
    already made, so a forward reference is a crash at spawn time."""
    for scene in load_scenes():
        seen = set()
        for obj in scene.objects:
            if obj.in_bowl:
                assert obj.kind == "cube", f"{scene.id}: only cubes can be in a bowl"
                assert obj.in_bowl in seen, \
                    f"{scene.id}: {obj.name} references {obj.in_bowl} before it exists"
            if obj.kind == "bowl":
                seen.add(obj.name)


def test_disturbance_blocks_are_parsed_and_land_inside_the_workspace():
    scenes = {s.id: s for s in load_scenes()}
    disturbed = {s.id for s in scenes.values() if s.disturbance}
    assert disturbed == {"disturb_h3", "disturb_match3"}
    for scene_id in disturbed:
        spec = scenes[scene_id].disturbance
        assert spec.action == "eject"
        assert spec.after_placements == 1
        x, y = spec.to
        assert WORKSPACE["x"][0] <= x <= WORKSPACE["x"][1], scene_id
        assert WORKSPACE["y"][0] <= y <= WORKSPACE["y"][1], scene_id
        # An eject that lands on a block's spawn point would collide rather than sit.
        for obj in scenes[scene_id].objects:
            if obj.kind == "cube":
                assert abs(obj.position[0] - x) > 0.05 or abs(obj.position[1] - y) > 0.05, \
                    f"{scene_id}: eject target sits on {obj.name}"


def test_failure_mode_and_disturbance_agree():
    for scene in load_scenes():
        assert (scene.failure_mode == "disturbance") == bool(scene.disturbance), scene.id


def test_no_cube_shares_a_colour_with_a_bowl_in_the_same_scene():
    """h4_quad paired a blue cube with a blue bowl. Held near the overhead camera the
    cube's blob exceeds the 5,000 px bowl threshold, is classified as a bowl, and
    steals the bowl's id -- leaving the real bowl unnameable for BOTH conditions. A
    perfect scripted policy fails such a scene, so it measures perception, not the
    agent. Scenes whose bowls are small and sparse survive it, so this is a warning
    the suite encodes rather than a law of physics -- but a NEW scene should not
    reintroduce it silently."""
    offenders = []
    for scene in load_scenes():
        cube_colours = {o.color for o in scene.objects if o.kind == "cube"}
        bowl_colours = {o.color for o in scene.objects if o.kind == "bowl"}
        # a clash is only fatal when the bowls are full-size (r >= 0.07): a big bowl
        # sets a high area bar that a held cube can cross.
        big = {o.color for o in scene.objects if o.kind == "bowl" and o.radius >= 0.07}
        fatal = cube_colours & big
        if fatal:
            offenders.append(f"{scene.id}: cube and full-size bowl share {sorted(fatal)}")
    assert offenders == [], "\n".join(offenders)
