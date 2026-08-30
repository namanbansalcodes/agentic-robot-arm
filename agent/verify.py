"""Three verification layers, three switches.

L1  loud errors        free      reads the error string a primitive already returned
L2  proprioception     free      reads the gripper aperture
L3  visual check       1 call    one narrow yes/no question about a fresh photo

Ordering is deliberate and is itself a finding: the free layers run first, and L3 is
only ever paid for when L1 and L2 have already said "looks fine". Most of the value
arrives before any extra token is spent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.llm import VLMCall
from agent.prompts import VERIFIER_SYSTEM, VERIFY_QUESTIONS
from primitives.api import EMPTY_GRIP_THRESHOLD
from primitives.feedback import Feedback


@dataclass(frozen=True)
class VerificationConfig:
    """Which layers are on. Three independent booleans, not a level dial.

    A single 0-3 "verification level" would make L2-without-L1 inexpressible, and the
    changelog needs a separately measured number for each layer -- including the
    deliberately-wasteful `verify_every_primitive` variant, which exists to price the
    naive "just check everything" policy rather than to be recommended.
    """

    l1: bool = True
    l2: bool = True
    l3: bool = True
    verify_every_primitive: bool = False

    @property
    def label(self) -> str:
        """The condition name. It becomes a cache key and a report row, so it is
        derived from the switches rather than passed in -- a hand-typed label can
        disagree with the config it names, and a cache key that lies is worse than
        no cache at all."""
        if self.verify_every_primitive:
            return "agent_verify_every"
        enabled = "".join(name for name, on in
                          (("L1", self.l1), ("L2", self.l2), ("L3", self.l3)) if on)
        return f"agent_{enabled}" if enabled else "agent_none"


@dataclass(frozen=True)
class Verdict:
    """ok, plus WHICH layer objected and why.

    The layer is recorded because "verification caught it" is not the finding -- the
    finding is which layer caught it, and how many of the catches were free.
    """

    ok: bool
    layer: Optional[str] = None
    reason: str = ""


class Verifier:
    """Runs the enabled layers, cheapest first, and stops at the first objection.

    `l3_calls` is the price tag: it counts the VLM calls verification itself spent,
    separately from the calls the policy spent on planning.
    """

    def __init__(self, config: VerificationConfig, client):
        self.config = config
        self.client = client
        self.l3_calls = 0

    def check(self, feedback: Feedback, subtask: Optional[str], scene,
              image_png: Optional[bytes], step_index: int) -> Verdict:
        """Cheapest layer first, and return on the first failure.

        Returning early is not just an optimisation, it is the claim under test: a
        run that had already been told "unreachable" for free and then paid for a
        photo to confirm it would price verification at whatever the wasteful
        ordering costs, not at what it actually costs.
        """
        if self.config.l1 and feedback.status == "error":
            return Verdict(False, "L1",
                           feedback.error or "primitive reported an error")

        if (self.config.l2 and feedback.primitive == "grasp"
                and feedback.status == "ok"
                and feedback.fingers_width < EMPTY_GRIP_THRESHOLD):
            return Verdict(
                False, "L2",
                f"gripper closed to {feedback.fingers_width:.4f} m -- it grasped air, "
                "the block is still on the table")

        # L3 is the only layer that costs anything, so it is gated twice: by a subtask
        # boundary (or the wasteful variant's explicit opt-in) and by having a frame.
        if (self.config.l3
                and (subtask is not None or self.config.verify_every_primitive)
                and scene is not None):
            question = self._question(subtask, scene)
            call = VLMCall(
                scene_id=getattr(scene, "id", "unknown"),
                condition=self.config.label,
                seed=getattr(scene, "_seed", 0),
                step_index=step_index,
                call_kind="verify",
                system=VERIFIER_SYSTEM,
                text=question,
                image_png=image_png,
                tools=[],
            )
            response = self.client.complete(call)
            self.l3_calls += 1
            # Anything that is not an opening "yes" is a failure. VERIFIER_SYSTEM asks
            # for one word and says to answer "no" when the photo does not settle it,
            # so a hedge is a no -- treating an unparseable answer as a pass would make
            # the layer report success exactly when it was least sure.
            if not response.text.strip().lower().startswith("yes"):
                return Verdict(False, "L3",
                               f"visual check failed: {question} -> "
                               f"{response.text.strip()}")

        return Verdict(True)

    @staticmethod
    def _question(subtask: Optional[str], scene) -> str:
        """One narrow yes/no question, naming both objects.

        Narrow questions get reliable answers; "assess the situation" does not. The
        ids are de-underscored so the question reads as English about the photo
        ("the red cube") rather than as a symbol the inspector must resolve.
        """
        item = getattr(scene.success, "item", "block").replace("_", " ")
        container = getattr(scene.success, "container", "bowl").replace("_", " ")
        template = VERIFY_QUESTIONS.get(subtask or "placed", VERIFY_QUESTIONS["placed"])
        return template.format(item=item, container=container)
