"""Every prompt in the project. Baseline and agent share the preamble verbatim --
that shared string IS the fairness guarantee, so do not fork it.

The comparison this project reports is only worth reading if the two conditions were
given the same job. They get the same model, the same tool schema, the same pixels,
the same scenes, and the same step budget; the ONLY difference is that one plans once
and executes blind while the other acts a primitive at a time and reads the feedback.
Copying PRIMITIVE_REFERENCE into a second string -- even identically -- would let the
two drift apart under later editing, and the whole result with them.
"""

PRIMITIVE_REFERENCE = """\
You control a Franka Panda arm on a tabletop, in simulation.

You do NOT get coordinates and you never produce coordinates. You refer to objects
only by the ids given in the detection list, e.g. "red_cube_1", "blue_bowl_1".
The control software computes every pose from the camera image.

Ids describe what is visible in the CURRENT photo and are re-derived every time you
call look(). They are not permanent names for objects.

Primitives:
  look()                       Retract the arm and take a fresh overhead photo.
                               Returns the current detection list.
  move_to(target_id)           Move the gripper above the named object.
  grasp(object_id)             Approach, close the gripper on the named block, lift.
  place(target_id)             Carry whatever is held over the named target and release.
  ask_human(question)          Ask the operator a question. Use ONLY when the
                               instruction genuinely underdetermines the goal.
  report_done(success, reason) End the episode and state whether the task succeeded.

Every primitive returns: status, an error string if it failed, the gripper aperture
in metres, the end-effector position, and a fresh detection list.

Gripper aperture is your most reliable signal about whether you are holding anything:
roughly 0.04-0.06 m means a block is between the fingers; below about 0.012 m the
fingers have closed on empty air.
"""

BASELINE_SYSTEM = PRIMITIVE_REFERENCE + """
You are planning in ONE shot. Study the photo, then emit the complete sequence of
primitive calls that accomplishes the instruction. The plan will be executed exactly
as written, without pausing, and you will not see the results. Emit the calls now.
"""

AGENT_SYSTEM = PRIMITIVE_REFERENCE + """
You work ONE primitive at a time. After each call you will be shown exactly what
happened: the status, any error, the gripper aperture, and a fresh photo.

Use that feedback. Specifically:
  * An error string tells you what went wrong -- change the plan, do not repeat the
    identical call and hope.
  * After a grasp, read the aperture. Near zero means you closed on air; the block
    is still on the table and you must try a different approach.
  * You will be shown a log of what you have already tried. Do not retry an approach
    that is recorded there as having failed.
  * If the instruction does not determine the goal -- for example it says "the bowl"
    and you can see two bowls -- call ask_human rather than guessing.

When you finish, call report_done. Report success=true ONLY if you have positive
evidence the task is actually complete. If you could not do it, say so: an honest
failure is a correct answer, and a false claim of success is the worst outcome
available to you.
"""

VERIFIER_SYSTEM = """\
You are a strict visual inspector for a robot arm. You will be shown one photo and
asked ONE yes/no question about it. Answer with exactly one word, "yes" or "no",
then on a second line give a short reason. If the photo does not let you tell,
answer "no".
"""


def planning_prompt(instruction: str, detections: list, memory_text: str,
                    last_feedback: str | None) -> str:
    seen = "\n".join(f"  - {d['id']} ({d['kind']}, {d['color']}, {d['where']})"
                     for d in detections) or "  (nothing detected)"
    parts = [f"Instruction: {instruction}", "", "Objects currently detected:", seen]
    if last_feedback:
        parts += ["", "Result of your last action:", last_feedback]
    if memory_text:
        parts += ["", "What you have already tried this episode:", memory_text]
    parts += ["", "Call exactly one primitive now."]
    return "\n".join(parts)


def baseline_prompt(instruction: str, detections: list) -> str:
    seen = "\n".join(f"  - {d['id']} ({d['kind']}, {d['color']}, {d['where']})"
                     for d in detections) or "  (nothing detected)"
    return "\n".join([f"Instruction: {instruction}", "", "Objects detected:", seen, "",
                      "Emit the full plan now as primitive calls."])


VERIFY_QUESTIONS = {
    "grasped": "Is the robot gripper holding the {item}, lifted clear of the table?",
    "placed": "Is the {item} inside the {container}?",
}
