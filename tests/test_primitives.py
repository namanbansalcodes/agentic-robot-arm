import numpy as np
import pytest

from primitives.api import EMPTY_GRIP_THRESHOLD, PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.world import World
from tests import SCENES, scene_with_human_answer, scene_with_unreachable


def _api(scene, seed=0, tmp_path=None):
    scene = SCENES[scene] if isinstance(scene, str) else scene
    world = World(scene, seed=seed)
    return world, PrimitiveAPI(RobotIO(world), image_dir=tmp_path)


def test_look_returns_detections_and_an_image(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    fb = api.look()
    assert fb.status == "ok"
    assert any(d["kind"] == "bowl" for d in fb.detections)
    assert fb.image_path and fb.image_path.endswith(".png")
    world.close()


def test_grasp_unknown_object_is_a_loud_error(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    api.look()
    fb = api.grasp("purple_cube_1")
    assert fb.status == "error"
    assert "unknown_object" in fb.error
    world.close()


def test_grasp_before_look_is_a_loud_error(tmp_path):
    """The VLM must call look() before it can name anything."""
    world, api = _api("h1_single", tmp_path=tmp_path)
    fb = api.grasp("red_cube_1")
    assert fb.status == "error" and "unknown_object" in fb.error
    world.close()


def test_grasping_a_bowl_is_rejected(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    api.look()
    fb = api.grasp("blue_bowl_1")
    assert fb.status == "error" and "bad_target" in fb.error
    world.close()


def test_grasp_outside_workspace_reports_unreachable(tmp_path):
    world, api = _api(scene_with_unreachable(SCENES["h1_single"], "red_cube"),
                      tmp_path=tmp_path)
    api.look()
    fb = api.grasp("red_cube_1")
    assert fb.status == "error"
    assert "unreachable" in fb.error
    world.close()


def test_grasping_air_closes_the_gripper_to_near_zero(tmp_path):
    """The L2 signal. If this does not separate, L2 cannot work."""
    world, api = _api("h1_single", tmp_path=tmp_path)
    api.look()
    empty = api._grasp_at(np.array([0.15, 0.15, 0.025]))
    assert empty.fingers_width < EMPTY_GRIP_THRESHOLD
    world.close()


def test_successful_grasp_holds_the_block(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    api.look()
    fb = api.grasp("red_cube_1")
    assert fb.status == "ok"
    assert fb.fingers_width > EMPTY_GRIP_THRESHOLD
    world.close()


def test_ask_human_records_the_answer(tmp_path):
    world, api = _api(scene_with_human_answer(SCENES["h1_single"],
                                             "Use the blue bowl."), tmp_path=tmp_path)
    fb = api.ask_human("Which bowl?")
    assert fb.status == "ok" and "blue" in fb.note.lower()
    world.close()


def test_report_done_records_the_claim(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    fb = api.report_done(False, "could not reach it")
    assert fb.status == "ok" and "success=False" in fb.note
    world.close()


def test_feedback_text_is_compact_and_mentions_the_aperture(tmp_path):
    world, api = _api("h1_single", tmp_path=tmp_path)
    text = api.look().to_model_text()
    assert "gripper_aperture_m" in text and "visible:" in text
    assert len(text.splitlines()) <= 8
    world.close()
