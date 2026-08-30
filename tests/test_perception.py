from harness.scenes import load_scenes
from primitives.perception import detect
from robotsim.io import RobotIO
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}


def _detect(scene_id, seed=0):
    world = World(SCENES[scene_id], seed=seed)
    io = RobotIO(world)
    io.retract()
    dets = detect(io.render("overhead"))
    world.close()
    return dets


def test_clean_scene_finds_one_cube_and_one_bowl():
    dets = _detect("clean_center")
    kinds = sorted((d.color, d.kind) for d in dets)
    assert ("blue", "bowl") in kinds
    assert ("red", "cube") in kinds


def test_ids_are_stable_across_repeated_detection():
    a = [d.id for d in _detect("distractor_three_cubes")]
    b = [d.id for d in _detect("distractor_three_cubes")]
    assert a == b and len(a) == len(set(a))


def test_three_cubes_are_separated():
    dets = _detect("distractor_three_cubes")
    cubes = {d.color for d in dets if d.kind == "cube"}
    assert cubes == {"red", "green", "yellow"}


def test_same_colour_bowl_and_cube_are_told_apart_by_size():
    """distractor_two_bowls is the trap: a red bowl AND a red cube."""
    dets = _detect("distractor_two_bowls")
    red = sorted((d.kind, d.area_px) for d in dets if d.color == "red")
    assert [k for k, _ in red] == ["bowl", "cube"]
    assert red[0][1] > 5000 and red[1][1] < 5000


def test_detection_carries_pixel_evidence_not_world_truth():
    d = _detect("clean_center")[0]
    assert 0 <= d.centroid_px[0] < 480 and 0 <= d.centroid_px[1] < 480
    assert d.area_px > 0
    assert isinstance(d.where, str) and d.where
