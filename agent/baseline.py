"""One VLM call. Full plan. Execute blind. Claim success if nothing crashed.

This is a fair, reasonable, and completely standard way to drive a VLM-controlled
arm -- which is exactly why it is the right baseline. It shares the model, the tool
schema, the primitives, the perception, the scenes, and the step budget with the
agent, and it reads the same PRIMITIVE_REFERENCE preamble. The ONLY difference in
the entire comparison is the loop.

The blindness is the experiment, not a bug. `claimed_success` starts True and is
lowered only if the model itself calls report_done: a plan that ran to the end
without the process crashing reports a job well done, whether or not the cube ever
left the table. That gap between what was claimed and what happened is the number
this project exists to measure, so nothing here may inspect a Feedback and change
course. No re-planning, no retries, no error handling between steps.

Scoring lives in harness/. Nothing in this file knows what "actually succeeded"
means -- it cannot, from behind the firewall, and that is the point.
"""
from __future__ import annotations

from agent.llm import VLMCall
from agent.prompts import BASELINE_SYSTEM, baseline_prompt
from agent.trace import EpisodeTrace
from primitives.api import PrimitiveAPI
from primitives.feedback import Feedback
from primitives.imaging import encode_png

# The single tool schema BOTH conditions use. Every parameter is a string or a
# boolean: there is no way to express a coordinate here, which is the firewall
# showing up in the model's own vocabulary rather than only in ours.
#
# move_to deliberately exposes target_id ONLY. PrimitiveAPI.move_to also accepts an
# `offset="at"` mode that descends to object height; handing that to the model got
# the gripper driven into a cube mid-episode. Withholding it costs the model nothing
# -- grasp() does its own approach -- so the schema stops at the safe surface.
TOOLS = [
    {"type": "function", "name": "look",
     "description": "Retract the arm and take a fresh overhead photo. Returns the "
                    "current detection list.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "move_to",
     "description": "Move the gripper above a named object.",
     "parameters": {"type": "object",
                    "properties": {"target_id": {
                        "type": "string",
                        "description": "Id of an object from the detection list."}},
                    "required": ["target_id"]}},
    {"type": "function", "name": "grasp",
     "description": "Approach, close the gripper on the named block, and lift.",
     "parameters": {"type": "object",
                    "properties": {"object_id": {
                        "type": "string",
                        "description": "Id of a block from the detection list."}},
                    "required": ["object_id"]}},
    {"type": "function", "name": "place",
     "description": "Carry whatever is held over the named target and release it.",
     "parameters": {"type": "object",
                    "properties": {"target_id": {
                        "type": "string",
                        "description": "Id of the target from the detection list."}},
                    "required": ["target_id"]}},
    {"type": "function", "name": "ask_human",
     "description": "Ask the operator a question. Use ONLY when the instruction "
                    "genuinely underdetermines the goal.",
     "parameters": {"type": "object",
                    "properties": {"question": {
                        "type": "string",
                        "description": "The question to ask the operator."}},
                    "required": ["question"]}},
    {"type": "function", "name": "report_done",
     "description": "End the episode and state whether the task succeeded.",
     "parameters": {"type": "object",
                    "properties": {
                        "success": {"type": "boolean",
                                    "description": "True only with positive evidence "
                                                   "the task is complete."},
                        "reason": {"type": "string",
                                   "description": "Short justification for the claim."}},
                    "required": ["success", "reason"]}},
]


def dispatch(api: PrimitiveAPI, name: str, args: dict) -> Feedback:
    """Tool name -> primitive call. The one place a model's word becomes a motion.

    Unknown names raise rather than being skipped: a silently dropped tool call would
    shorten the plan without leaving a trace of having done so, which is precisely the
    kind of invisible help that would make the baseline unfair to compare against.
    """
    if name == "look":
        return api.look()
    if name == "move_to":
        return api.move_to(args["target_id"])
    if name == "grasp":
        return api.grasp(args["object_id"])
    if name == "place":
        return api.place(args["target_id"])
    if name == "ask_human":
        return api.ask_human(args["question"])
    if name == "report_done":
        return api.report_done(bool(args.get("success", True)), args.get("reason", ""))
    raise KeyError(name)


def plan_once(scene, seed: int, io, api, client, tools, dispatch) -> EpisodeTrace:
    """Plan the whole episode in one VLM call, then execute it without looking.

    `io`, `api`, `client`, `tools` and `dispatch` are injected so the harness owns the
    simulator and this module stays a policy: it decides what to call, never what the
    result meant.
    """
    first = api.look()
    trace = EpisodeTrace(steps=[{"primitive": "look", "args": {}, "reasoning": "",
                                 "feedback": first.to_dict()}])

    call = VLMCall(
        scene_id=scene.id, condition="baseline", seed=seed, step_index=0,
        call_kind="baseline_plan", system=BASELINE_SYSTEM,
        text=baseline_prompt(scene.instruction, first.detections),
        image_png=encode_png(io.render("overhead")), tools=tools,
    )
    response = client.complete(call)   # the only VLM call in the entire episode

    # A plan that finished is a plan that succeeded, as far as this policy can tell.
    trace.claimed_success = True
    trace.claim_reason = "the one-shot plan ran to completion"
    trace.stop_reason = "executed the one-shot plan"

    for tool_call in response.tool_calls[: scene.max_steps]:
        args = tool_call.get("args", {}) or {}
        feedback = dispatch(api, tool_call["name"], args)
        trace.steps.append({"primitive": tool_call["name"], "args": args,
                            "reasoning": response.text, "feedback": feedback.to_dict()})
        # Everything below reads the model's OWN call, never the feedback it produced.
        if tool_call["name"] == "ask_human":
            trace.escalations += 1
        if tool_call["name"] == "report_done":
            trace.claimed_success = bool(args.get("success", True))
            trace.claim_reason = str(args.get("reason", ""))
            trace.stop_reason = "agent called report_done"

    return trace
