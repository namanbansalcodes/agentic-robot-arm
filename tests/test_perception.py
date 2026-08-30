from primitives.perception import detect
from robotsim.io import RobotIO
from robotsim.world import World
from tests import SCENES


def _detect(scene_id, seed=0):
    world = World(SCENES[scene_id], seed=seed)
    io = RobotIO(world)
    io.retract()
    dets = detect(io.render("overhead"))
    world.close()
    return dets


def test_clean_scene_finds_one_cube_and_one_bowl():
    dets = _detect("h1_single")
    kinds = sorted((d.color, d.kind) for d in dets)
    assert ("blue", "bowl") in kinds
    assert ("red", "cube") in kinds


def test_ids_are_stable_across_repeated_detection():
    a = [d.id for d in _detect("h3_triple")]
    b = [d.id for d in _detect("h3_triple")]
    assert a == b and len(a) == len(set(a))


def test_three_cubes_are_separated():
    dets = _detect("h3_triple")
    cubes = {d.color for d in dets if d.kind == "cube"}
    assert cubes == {"red", "green", "yellow"}


def test_a_small_bowl_still_reads_as_a_bowl():
    """The three-bowl scenes shrink the bowls to r=0.060 to fit. If that dropped a
    bowl under the 5,000 px threshold it would be classified as a CUBE, and the agent
    would be told to pick up its own target."""
    dets = _detect("match3")
    bowls = [d for d in dets if d.kind == "bowl"]
    assert len(bowls) == 3
    assert all(d.area_px > 5000 for d in bowls), [d.area_px for d in bowls]


def test_a_cube_and_a_bowl_of_the_same_colour_get_distinct_ids():
    """h4_quad puts a blue cube in front of a blue bowl. The id counter is keyed on
    colour AND kind, so they must not collide."""
    ids = {d.id for d in _detect("h4_quad")}
    assert "blue_cube_1" in ids and "blue_bowl_1" in ids


def test_same_colour_bowl_and_cube_are_told_apart_by_size():
    """match3 is the trap: a red bowl AND a red cube, in the same frame."""
    dets = _detect("match3")
    red = sorted((d.kind, d.area_px) for d in dets if d.color == "red")
    assert [k for k, _ in red] == ["bowl", "cube"]
    assert red[0][1] > 5000 and red[1][1] < 5000


def test_detection_carries_pixel_evidence_not_world_truth():
    d = _detect("h1_single")[0]
    assert 0 <= d.centroid_px[0] < 480 and 0 <= d.centroid_px[1] < 480
    assert d.area_px > 0
    assert isinstance(d.where, str) and d.where
