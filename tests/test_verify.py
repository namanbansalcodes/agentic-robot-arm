import pytest

from agent.llm import VLMResponse
from agent.verify import VerificationConfig, Verifier, Verdict
from primitives.feedback import Feedback


class StubClient:
    def __init__(self, answer="yes"):
        self.answer, self.calls = answer, 0
        self.input_tokens = self.output_tokens = self.drift_count = 0

    def complete(self, call):
        self.calls += 1
        self.last_call = call
        return VLMResponse(text=self.answer, tool_calls=[], input_tokens=1,
                           output_tokens=1, model="stub")

    def cost_usd(self):
        return 0.0


class _Scene:
    id = "clean_center"
    _seed = 0
    instruction = "Put the red block in the blue bowl."
    success = type("S", (), {"item": "red_cube", "container": "blue_bowl"})()


def _fb(**kw):
    base = dict(primitive="grasp", args={"object_id": "red_cube_1"}, status="ok",
                fingers_width=0.048, ee_position=(0, 0, 0.2), detections=[])
    base.update(kw)
    return Feedback(**base)


def test_all_layers_off_never_objects():
    v = Verifier(VerificationConfig(l1=False, l2=False, l3=False), StubClient())
    verdict = v.check(_fb(status="error", error="unreachable: nope"), subtask="grasped",
                      scene=_Scene(), image_png=b"", step_index=0)
    assert verdict.ok is True and verdict.layer is None


def test_l1_catches_an_error_string():
    v = Verifier(VerificationConfig(l1=True, l2=False, l3=False), StubClient())
    verdict = v.check(_fb(status="error", error="unreachable: outside workspace"),
                      subtask="grasped", scene=_Scene(), image_png=b"", step_index=0)
    assert verdict.ok is False and verdict.layer == "L1"
    assert "unreachable" in verdict.reason


def test_l2_catches_an_empty_gripper_that_l1_reported_as_ok():
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=False), StubClient())
    verdict = v.check(_fb(status="ok", fingers_width=0.003), subtask="grasped",
                      scene=_Scene(), image_png=b"", step_index=0)
    assert verdict.ok is False and verdict.layer == "L2"
    assert "air" in verdict.reason.lower()


def test_l2_costs_no_vlm_calls():
    stub = StubClient()
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=False), stub)
    v.check(_fb(fingers_width=0.003), subtask="grasped", scene=_Scene(),
            image_png=b"", step_index=0)
    assert stub.calls == 0, "proprioception must be free -- that is its whole advantage"


def test_l2_only_fires_on_grasp():
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=False), StubClient())
    verdict = v.check(_fb(primitive="place", fingers_width=0.003), subtask=None,
                      scene=_Scene(), image_png=b"", step_index=0)
    assert verdict.ok is True


def test_l3_does_not_run_without_a_subtask_boundary():
    stub = StubClient(answer="no")
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)
    v.check(_fb(primitive="move_to", status="ok"), subtask=None,
            scene=_Scene(), image_png=b"png", step_index=0)
    assert stub.calls == 0, "no boundary, no visual check"


def test_l3_runs_at_a_boundary_and_can_fail():
    stub = StubClient(answer="no\nthe bowl is empty")
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)
    verdict = v.check(_fb(primitive="place", status="ok", fingers_width=0.070),
                      subtask="placed", scene=_Scene(), image_png=b"png", step_index=1)
    assert stub.calls == 1
    assert verdict.ok is False and verdict.layer == "L3"
    assert v.l3_calls == 1


def test_l3_yes_passes():
    stub = StubClient(answer="yes\nthe block is clearly inside")
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)
    verdict = v.check(_fb(primitive="place", status="ok", fingers_width=0.070),
                      subtask="placed", scene=_Scene(), image_png=b"png", step_index=1)
    assert verdict.ok is True


def test_l3_is_never_paid_for_when_a_free_layer_already_failed():
    """The ordering IS the finding: never spend a token to confirm a known failure."""
    stub = StubClient()
    v = Verifier(VerificationConfig(l1=True, l2=True, l3=True), stub)
    v.check(_fb(status="error", error="unreachable"), subtask="grasped",
            scene=_Scene(), image_png=b"png", step_index=0)
    v.check(_fb(status="ok", fingers_width=0.001), subtask="grasped",
            scene=_Scene(), image_png=b"png", step_index=1)
    assert stub.calls == 0


def test_l3_asks_a_narrow_question_naming_both_objects():
    stub = StubClient()
    v = Verifier(VerificationConfig(l3=True), stub)
    v.check(_fb(primitive="place", status="ok", fingers_width=0.07), subtask="placed",
            scene=_Scene(), image_png=b"png", step_index=1)
    q = stub.last_call.text.lower()
    assert "red cube" in q and "blue bowl" in q
    assert q.count("?") == 1, "one narrow question, not an open-ended assessment"


def test_verify_every_primitive_runs_l3_without_a_boundary():
    stub = StubClient()
    v = Verifier(VerificationConfig(verify_every_primitive=True), stub)
    v.check(_fb(primitive="move_to", status="ok"), subtask=None,
            scene=_Scene(), image_png=b"png", step_index=0)
    assert stub.calls == 1


@pytest.mark.parametrize("cfg,label", [
    (VerificationConfig(True, True, True), "agent_L1L2L3"),
    (VerificationConfig(True, False, False), "agent_L1"),
    (VerificationConfig(True, True, False), "agent_L1L2"),
    (VerificationConfig(False, False, False), "agent_none"),
    (VerificationConfig(True, True, True, verify_every_primitive=True), "agent_verify_every"),
])
def test_labels_are_stable_condition_names(cfg, label):
    """These labels become cache keys and report rows -- they must not drift."""
    assert cfg.label == label


def test_verify_every_non_boundary_failure_is_informational_not_a_failure():
    """The removed-experiment must pay the token cost without being a strawman.
    "Is the cube in the bowl?" is correctly "no" after a move_to, and treating that
    as a failure would make the agent thrash against its own verifier."""
    stub = StubClient(answer="no\nthe cube is on the table")
    v = Verifier(VerificationConfig(verify_every_primitive=True), stub)
    verdict = v.check(_fb(primitive="move_to", status="ok"), subtask=None,
                      scene=_Scene(), image_png=b"png", step_index=0)
    assert stub.calls == 1, "it must still cost a call -- that is what we are pricing"
    assert verdict.ok is True and verdict.informational is True


def test_verify_every_still_fails_hard_at_a_real_boundary():
    stub = StubClient(answer="no\nthe bowl is empty")
    v = Verifier(VerificationConfig(verify_every_primitive=True), stub)
    verdict = v.check(_fb(primitive="place", status="ok", fingers_width=0.07),
                      subtask="placed", scene=_Scene(), image_png=b"png", step_index=1)
    assert verdict.ok is False and verdict.layer == "L3" and verdict.informational is False
