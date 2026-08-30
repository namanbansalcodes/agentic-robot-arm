import numpy as np

from agent.baseline import TOOLS, dispatch, plan_once
from agent.llm import VLMResponse
from harness.scenes import load_scenes
from primitives.api import PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.world import World

SCENES = {s.id: s for s in load_scenes()}


class StubClient:
    """Same interface as LLMClient, canned answers. Tests the LOOP, not the model."""
    def __init__(self, responses):
        self._responses, self.i = responses, 0
        self.input_tokens = self.output_tokens = self.calls = self.drift_count = 0

    def complete(self, call):
        r = self._responses[min(self.i, len(self._responses) - 1)]
        self.i += 1
        self.calls += 1
        return r

    def cost_usd(self):
        return 0.0


def _resp(tool_calls):
    return VLMResponse(text="", tool_calls=tool_calls, input_tokens=5,
                       output_tokens=5, model="stub")


def _run(scene_id, tool_calls, tmp_path):
    world = World(SCENES[scene_id], seed=0)
    io = RobotIO(world)
    api = PrimitiveAPI(io, image_dir=tmp_path)
    client = StubClient([_resp(tool_calls)])
    trace = plan_once(SCENES[scene_id], 0, io, api, client, TOOLS, dispatch)
    world.close()
    return trace, client


def test_baseline_makes_exactly_one_planning_call(tmp_path):
    trace, client = _run("clean_center", [
        {"name": "grasp", "args": {"object_id": "red_cube_1"}},
        {"name": "place", "args": {"target_id": "blue_bowl_1"}},
    ], tmp_path)
    assert client.calls == 1, "the baseline plans once and executes blind"
    assert trace.claimed_success is True
    assert len(trace.steps) >= 3


def test_baseline_still_claims_success_after_a_failing_step(tmp_path):
    """The behaviour this whole project exists to measure. Do not 'fix' it."""
    trace, _ = _run("clean_center", [
        {"name": "grasp", "args": {"object_id": "nonexistent_1"}},
        {"name": "place", "args": {"target_id": "blue_bowl_1"}},
    ], tmp_path)
    assert any(s["feedback"]["status"] == "error" for s in trace.steps)
    assert trace.claimed_success is True


def test_baseline_never_reads_feedback_between_steps(tmp_path):
    """Blind execution: an unreachable grasp does not stop the rest of the plan."""
    trace, _ = _run("unreachable_block", [
        {"name": "grasp", "args": {"object_id": "red_cube_1"}},
        {"name": "place", "args": {"target_id": "blue_bowl_1"}},
    ], tmp_path)
    prims = [s["primitive"] for s in trace.steps]
    assert "place" in prims, "the baseline must keep going after an error"
    assert trace.claimed_success is True


def test_baseline_honours_its_own_report_done(tmp_path):
    trace, _ = _run("clean_center", [
        {"name": "report_done", "args": {"success": False, "reason": "gave up"}},
    ], tmp_path)
    assert trace.claimed_success is False
    assert "gave up" in trace.claim_reason


def test_tool_schema_exposes_no_coordinates_and_no_offset(tmp_path):
    names = {t["name"] for t in TOOLS}
    assert names == {"look", "move_to", "grasp", "place", "ask_human", "report_done"}
    for tool in TOOLS:
        props = tool["parameters"].get("properties", {})
        for pname, spec in props.items():
            assert spec["type"] in ("string", "boolean"), f"{tool['name']}.{pname} is not symbolic"
        assert "offset" not in props, "move_to must not expose the colliding 'at' offset"


def test_both_conditions_share_one_preamble():
    from agent.prompts import AGENT_SYSTEM, BASELINE_SYSTEM, PRIMITIVE_REFERENCE
    assert PRIMITIVE_REFERENCE in BASELINE_SYSTEM
    assert PRIMITIVE_REFERENCE in AGENT_SYSTEM
    assert len(PRIMITIVE_REFERENCE) > 400
