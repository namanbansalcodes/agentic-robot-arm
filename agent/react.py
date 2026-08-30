"""ReAct loop: observe -> think -> ONE primitive -> read feedback -> verify -> decide.

Budgets are hard. Blowing one is an honest FAILURE verdict -- never a hang, and never
a claimed success.

This is the whole contribution of the project, and it is deliberately small. It shares
the model, the tool schema, the dispatcher, the primitives, the perception, the scenes
and the step budget with agent/baseline.py; the only difference in the entire comparison
is that this file reads the Feedback each primitive returned and gets to change its mind.
There is no planning ahead, no tree search, and no second model -- every one of those
would add a second variable to a two-condition experiment and cost us the ability to say
what the loop itself bought.

Nothing here knows what "actually succeeded" means. `claimed_success` is the agent's
own claim, scored against the oracle in harness/, on the other side of the firewall.
"""
from __future__ import annotations

from agent.llm import VLMCall
from agent.memory import EpisodeMemory
from agent.prompts import AGENT_SYSTEM, planning_prompt
from agent.trace import EpisodeTrace
from agent.verify import Verifier
from primitives.feedback import Feedback
from primitives.imaging import encode_png

# Which primitives close out a subtask, and therefore what the L3 visual check should
# ask about. Everything not named here is a means, not an end: a move_to has no
# "did it work" question worth paying a VLM call for, and asking one anyway is the
# wasteful policy VerificationConfig.verify_every_primitive exists to price.
def _verify_frame(io, config) -> bytes:
    """Only rasterise the oblique view when a layer will actually look at it.

    The ablation conditions run with l3=False; rendering a frame nobody reads cost
    simulator time on every acting step of every episode, which across 250 episodes
    is minutes of wall clock for nothing.
    """
    if not config.l3:
        return b""
    return encode_png(io.render("oblique"))

SUBTASK_OF = {"grasp": "grasped", "place": "placed"}


def run_agent_policy(scene, seed: int, io, api, client, config, tools,
                     dispatch) -> EpisodeTrace:
    """Run one closed-loop episode and return its trace.

    `io`, `api`, `client`, `tools` and `dispatch` are injected for the same reason as
    in the baseline: the harness owns the simulator, and this module stays a policy.
    `tools` and `dispatch` are the baseline's own objects, passed in rather than
    redeclared -- a second tool schema that drifted from the first would quietly turn
    the comparison into a comparison of prompts.
    """
    verifier = Verifier(config, client, seed=seed)
    memory = EpisodeMemory()
    trace = EpisodeTrace()

    # Step 0 is free and is not charged against the budget: an agent that has not yet
    # looked has no ids to name, so making it spend a decision step on the first photo
    # would hand the baseline a step it does not have to pay for either.
    feedback = api.look()
    trace.steps.append({"primitive": "look", "args": {}, "reasoning": "",
                        "feedback": feedback.to_dict(), "verdict": None})
    memory.record("look", {}, "ok")

    consecutive_failures = 0

    for step_index in range(1, scene.max_steps + 1):
        call = VLMCall(
            scene_id=scene.id, condition=config.label, seed=seed,
            step_index=step_index, call_kind="plan", system=AGENT_SYSTEM,
            # Three things go in, and the middle one is the fix this task exists for:
            # the raw last result, the attempt log that outlives it, and the photo.
            text=planning_prompt(scene.instruction, feedback.detections,
                                 memory.as_text(), feedback.to_model_text()),
            image_png=encode_png(io.render("overhead")), tools=tools,
        )
        response = client.complete(call)

        if not response.tool_calls:
            trace.stop_reason = "model produced no primitive call"
            break

        # ONE primitive per step, always. A turn carrying several calls is the model
        # trying to go open-loop; honouring the extras would make this the baseline
        # wearing a loop's clothes, and the feedback it never read would be the
        # difference we were supposed to be measuring.
        tool_call = response.tool_calls[0]
        name = tool_call.get("name")
        args = tool_call.get("args", {}) or {}

        if name == "report_done":
            trace.claimed_success = bool(args.get("success", False))
            trace.claim_reason = str(args.get("reason", ""))
            trace.stop_reason = "agent called report_done"
            # Execute it anyway: report_done takes a frame, so the claim lands in the
            # transcript next to the photo it was made about.
            feedback = api.report_done(trace.claimed_success, trace.claim_reason)
            trace.steps.append({"primitive": name, "args": args,
                                "reasoning": response.text,
                                "feedback": feedback.to_dict(), "verdict": None})
            break

        if name == "ask_human":
            trace.escalations += 1

        try:
            feedback = dispatch(api, name, args)
        except KeyError:
            # A hallucinated tool name ends one step, not the episode. Crashing here
            # would score a model typo as a harness failure, and silently skipping it
            # would leave the agent staring at feedback from a call it never made.
            feedback = Feedback(
                primitive=str(name), args=args, status="error",
                error=f"unknown_primitive: '{name}' is not one of the six primitives",
            )

        subtask = SUBTASK_OF.get(name)
        verdict = verifier.check(feedback, subtask=subtask, scene=scene,
                                 image_png=_verify_frame(io, config),
                                 step_index=step_index)

        # An informational verdict is already ok=True; recording its reason keeps the
        # observation the run paid for without letting a question that was never
        # meaningful at this step count as a failure.
        outcome = "ok" if verdict.ok else f"failed: {verdict.reason}"
        if verdict.informational:
            outcome = f"ok ({verdict.reason})"
        memory.record(name, args, outcome)

        trace.steps.append({"primitive": name, "args": args,
                            "reasoning": response.text,
                            "feedback": feedback.to_dict(),
                            "verdict": {"ok": verdict.ok, "layer": verdict.layer,
                                        "reason": verdict.reason,
                                        "informational": verdict.informational}})

        if verdict.ok:
            consecutive_failures = 0
            continue

        trace.recoveries += 1
        consecutive_failures += 1
        if consecutive_failures >= scene.max_retries_per_subtask:
            # Out of retries is a failure the agent must own. Saying so is the correct
            # answer here -- unreachable_block is a scene where it is the ONLY correct
            # answer -- and a false claim is the worst outcome available.
            trace.claimed_success = False
            trace.claim_reason = (
                f"{consecutive_failures} consecutive failed attempts; "
                f"last failure: {verdict.reason}")
            trace.stop_reason = "retry budget exhausted"
            break
    else:
        # The budget ran out with the job unverified. There is no branch anywhere in
        # this loop that exhausts a budget and leaves claimed_success True.
        trace.claimed_success = False
        trace.claim_reason = "step budget exhausted before the task was verified complete"
        trace.stop_reason = "step budget exhausted"

    trace.l3_calls = verifier.l3_calls
    if not trace.stop_reason:
        trace.stop_reason = "episode ended without an explicit stop reason"
    return trace
