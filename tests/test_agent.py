"""The ReAct loop and its episode memory, driven by a scripted stub client.

These tests exercise the LOOP, not the model. Every VLM answer is canned, so what is
under test is the control flow the loop imposes on whatever the model says: one
primitive per step, hard budgets, an honest claim when a budget runs out, and a memory
that survives longer than one turn.
"""
import pytest

from agent.baseline import TOOLS, dispatch
from agent.llm import VLMResponse
from agent.memory import EpisodeMemory
from agent.react import run_agent_policy
from agent.verify import VerificationConfig
from primitives.api import PrimitiveAPI
from robotsim.io import RobotIO
from robotsim.world import World
from tests import SCENES, scene_with_human_answer, scene_with_unreachable


class ScriptedClient:
    """Same interface as LLMClient. Planning calls get the next scripted turn;
    verification calls always answer "yes".

    Splitting on call_kind is not cosmetic. If a verifier's yes/no consumed a script
    entry, every later step would silently shift by one and the test would be asserting
    about a plan nobody wrote. The stub also records every planning prompt, so a test
    can inspect what the loop actually put in front of the model.
    """

    def __init__(self, script, verify_answer="yes"):
        self.script = [list(turn) for turn in script]
        self.verify_answer = verify_answer
        self.prompts = []                 # every planning prompt text, in order
        self.plan_calls = 0
        self.verify_calls = 0
        self.calls = 0
        self.input_tokens = self.output_tokens = self.drift_count = 0

    def complete(self, call):
        self.calls += 1
        if call.call_kind == "verify":
            self.verify_calls += 1
            return VLMResponse(text=self.verify_answer, tool_calls=[], input_tokens=1,
                               output_tokens=1, model="stub")
        self.prompts.append(call.text)
        turn = self.script[min(self.plan_calls, len(self.script) - 1)]
        self.plan_calls += 1
        return VLMResponse(text="", tool_calls=turn, input_tokens=5, output_tokens=5,
                           model="stub")

    def cost_usd(self):
        return 0.0


def _run(scene_id, script, tmp_path, config=None, verify_answer="yes"):
    scene = SCENES[scene_id] if isinstance(scene_id, str) else scene_id
    w = World(scene, seed=0)
    io = RobotIO(w)
    api = PrimitiveAPI(io, image_dir=tmp_path)
    client = ScriptedClient(script, verify_answer=verify_answer)
    try:
        trace = run_agent_policy(scene, 0, io, api, client,
                                 config or VerificationConfig(), TOOLS, dispatch)
    finally:
        w.close()
    return trace, client


def _prims(trace):
    return [s["primitive"] for s in trace.steps]


# --- the loop ----------------------------------------------------------------

def test_agent_completes_a_clean_scene(tmp_path):
    trace, _ = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "red_cube_1"}}],
        [{"name": "place", "args": {"target_id": "blue_bowl_1"}}],
        [{"name": "report_done", "args": {"success": True, "reason": "cube is in the bowl"}}],
    ], tmp_path)
    assert trace.claimed_success is True
    assert trace.stop_reason == "agent called report_done"
    assert _prims(trace) == ["look", "grasp", "place", "report_done"]


def test_agent_never_claims_success_after_exhausting_the_step_budget(tmp_path):
    """The core honesty guarantee: no path exists where a budget runs out and the
    agent still claims it finished the job."""
    scene = SCENES["h1_single"]
    trace, _ = _run("h1_single", [[{"name": "look", "args": {}}]], tmp_path)
    assert trace.claimed_success is False
    assert "budget" in trace.claim_reason
    assert trace.stop_reason == "step budget exhausted"
    # step 0's look, plus one per budgeted step -- and not one more.
    assert len(trace.steps) == scene.max_steps + 1


def test_agent_stops_after_the_retry_budget(tmp_path):
    scene = scene_with_unreachable(SCENES["h1_single"], "red_cube")
    trace, _ = _run(scene,
                    [[{"name": "grasp", "args": {"object_id": "red_cube_1"}}]], tmp_path)
    assert trace.claimed_success is False
    assert trace.recoveries >= 1
    assert trace.stop_reason == "retry budget exhausted"
    grasps = [p for p in _prims(trace) if p == "grasp"]
    assert len(grasps) <= scene.max_retries_per_subtask
    assert len(trace.steps) < scene.max_steps + 1, "the retry budget must bite first"


def test_agent_records_escalation(tmp_path):
    trace, _ = _run(scene_with_human_answer(SCENES["h1_single"], "Use the blue bowl."), [
        [{"name": "ask_human", "args": {"question": "Which bowl?"}}],
        [{"name": "report_done", "args": {"success": False, "reason": "asked, then stopped"}}],
    ], tmp_path)
    assert trace.escalations == 1
    assert "ask_human" in _prims(trace)


def test_agent_executes_exactly_one_primitive_per_step(tmp_path):
    """Two calls in one turn is the model trying to go open-loop. The loop takes the
    first and throws the rest away -- otherwise the agent quietly becomes the baseline."""
    trace, _ = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "red_cube_1"}},
         {"name": "place", "args": {"target_id": "blue_bowl_1"}}],
        [{"name": "report_done", "args": {"success": False, "reason": "stopping"}}],
    ], tmp_path)
    assert _prims(trace) == ["look", "grasp", "report_done"]
    assert "place" not in _prims(trace)


def test_agent_stops_when_the_model_calls_no_primitive(tmp_path):
    trace, _ = _run("h1_single", [[]], tmp_path)
    assert trace.claimed_success is False
    assert trace.stop_reason == "model produced no primitive call"


def test_memory_accumulates_failures(tmp_path):
    """The measured before-case in one assertion: what happened on step N must still
    be visible on step N+2, long after the raw feedback has scrolled away."""
    _, client = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "no_such_cube_1"}}],
        [{"name": "look", "args": {}}],
        [{"name": "report_done", "args": {"success": False, "reason": "stopping"}}],
    ], tmp_path)
    assert len(client.prompts) >= 3
    assert "failed" not in client.prompts[0], "nothing has failed yet on the first turn"
    for prompt in client.prompts[1:3]:
        assert "no_such_cube_1" in prompt
        assert "failed" in prompt


def test_memory_survives_more_than_one_turn(tmp_path):
    """The ablation, as a unit test: the answer to an ask_human is still in context
    two steps later. Without memory it was gone after one."""
    _, client = _run(scene_with_human_answer(SCENES["h1_single"], "Use the blue bowl."), [
        [{"name": "ask_human", "args": {"question": "Which bowl should I use?"}}],
        [{"name": "look", "args": {}}],
        [{"name": "look", "args": {}}],
        [{"name": "report_done", "args": {"success": False, "reason": "stopping"}}],
    ], tmp_path)
    assert "ask_human" in client.prompts[-1], (
        "the escalation must still be visible in the last prompt of the episode")


def test_unknown_tool_name_does_not_crash_the_episode(tmp_path):
    trace, _ = _run("h1_single", [
        [{"name": "teleport", "args": {}}],
        [{"name": "report_done", "args": {"success": False, "reason": "stopping"}}],
    ], tmp_path)
    assert "teleport" in _prims(trace)
    bad = next(s for s in trace.steps if s["primitive"] == "teleport")
    assert bad["feedback"]["status"] == "error"
    assert "teleport" in (bad["feedback"]["error"] or "")
    assert trace.stop_reason == "agent called report_done"


def test_every_acting_step_carries_a_verdict(tmp_path):
    trace, _ = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "red_cube_1"}}],
        [{"name": "report_done", "args": {"success": True, "reason": "held it"}}],
    ], tmp_path)
    grasp_step = next(s for s in trace.steps if s["primitive"] == "grasp")
    assert set(grasp_step["verdict"]) == {"ok", "layer", "reason", "informational"}


def test_l3_calls_are_counted_on_the_trace(tmp_path):
    trace, client = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "red_cube_1"}}],
        [{"name": "report_done", "args": {"success": True, "reason": "held it"}}],
    ], tmp_path)
    assert trace.l3_calls == client.verify_calls


def test_layers_off_means_no_verification_calls(tmp_path):
    trace, client = _run("h1_single", [
        [{"name": "grasp", "args": {"object_id": "red_cube_1"}}],
        [{"name": "report_done", "args": {"success": True, "reason": "held it"}}],
    ], tmp_path, config=VerificationConfig(l1=False, l2=False, l3=False))
    assert client.verify_calls == 0
    assert trace.l3_calls == 0
    assert trace.recoveries == 0


def test_stop_reason_is_never_empty(tmp_path):
    trace, _ = _run("h1_single", [
        [{"name": "report_done", "args": {"success": True, "reason": "done"}}],
    ], tmp_path)
    assert trace.stop_reason


# --- episode memory ----------------------------------------------------------

def test_memory_records_and_renders():
    m = EpisodeMemory()
    m.record("grasp", {"object_id": "red_cube_1"}, "ok")
    m.record("place", {"target_id": "blue_bowl_1"}, "failed: unreachable")
    text = m.as_text()
    assert "1. grasp(object_id='red_cube_1') -> ok" in text
    assert "2. place(target_id='blue_bowl_1') -> failed: unreachable" in text


def test_memory_starts_empty():
    assert EpisodeMemory().as_text() == ""


def test_has_tried_only_matches_failures():
    m = EpisodeMemory()
    m.record("grasp", {"object_id": "red_cube_1"}, "ok")
    assert m.has_tried("grasp", {"object_id": "red_cube_1"}) is False, (
        "a call that worked is not a call to avoid repeating")
    m.record("grasp", {"object_id": "far_cube_1"}, "failed: unreachable")
    assert m.has_tried("grasp", {"object_id": "far_cube_1"}) is True
    assert m.has_tried("grasp", {"object_id": "other_cube_1"}) is False
    assert m.has_tried("place", {"target_id": "far_cube_1"}) is False


def test_memory_limit_caps_the_injected_text():
    m = EpisodeMemory()
    for i in range(20):
        m.record("look", {"n": i}, "ok")
    lines = m.as_text(limit=5).splitlines()
    assert len(lines) == 5
    assert "n=19" in lines[-1], "the cap must keep the most RECENT entries"
    assert "n=15" in lines[0]
    assert len(m.entries) == 20, "the cap is on what is shown, not on what is kept"
