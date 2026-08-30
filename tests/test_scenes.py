from harness.scenes import load_scenes

REQUIRED_FAILURE_MODES = {
    "none", "perception", "hard_grasp", "occlusion", "ambiguity", "planner_error",
}


def test_loads_ten_scenes():
    scenes = load_scenes()
    assert len(scenes) == 10
    assert len({s.id for s in scenes}) == 10, "scene ids must be unique"


def test_every_failure_mode_is_covered():
    modes = {s.failure_mode for s in load_scenes()}
    assert modes == REQUIRED_FAILURE_MODES


def test_defaults_are_applied():
    scene = {s.id: s for s in load_scenes()}["clean_center"]
    assert scene.max_steps == 14
    cube = [o for o in scene.objects if o.kind == "cube"][0]
    assert cube.half_extent == 0.025
    assert cube.lateral_friction == 2.0


def test_ambiguous_scene_carries_a_human_answer():
    scene = {s.id: s for s in load_scenes()}["ambiguous_two_bowls"]
    assert scene.human_answer == "Use the blue bowl."
    assert scene.success.type == "contained_after_asking"
