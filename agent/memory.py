"""In-context attempt log. No vector DB, no external store -- that would be scope creep.

Measured before-case: without this, an agent that asked the operator "which bowl?"
and was told "the blue bowl" then placed in yellow, blue, yellow -- burning its whole
budget -- because the answer scrolled out of context after one step. It was never
confused about the goal; it simply could no longer see the answer it had asked for.

The fix is a list of strings. That is the whole mechanism, and it is deliberately the
whole mechanism: the failure was not a retrieval problem, so retrieval would have been
machinery bolted on to a bug it does not touch. What the episode needs is that the last
dozen things it did stay in front of it, in the order it did them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMemory:
    """One episode's attempt log. Created per episode and thrown away with it --
    nothing here persists across episodes, so no run can be contaminated by another."""

    entries: list = field(default_factory=list)

    def record(self, primitive: str, args: dict, outcome: str) -> None:
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self.entries.append(f"{primitive}({arg_text}) -> {outcome}")

    def has_tried(self, primitive: str, args: dict) -> bool:
        """Has this EXACT call already been recorded as having failed?

        Failures only. A call that worked is not a call to avoid repeating -- look()
        succeeds every time and must stay available -- and the arguments are part of
        the identity, because grasping a different cube is a different attempt.
        """
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        prefix = f"{primitive}({arg_text}) -> "
        return any(e.startswith(prefix) and "failed" in e for e in self.entries)

    def as_text(self, limit: int = 12) -> str:
        """The last `limit` entries, numbered, for injection into the next prompt.

        The cap is on what is SHOWN, not on what is kept: entries stays complete so
        has_tried can still see an early failure that has scrolled off the display.
        """
        return "\n".join(f"  {i+1}. {e}" for i, e in enumerate(self.entries[-limit:]))
