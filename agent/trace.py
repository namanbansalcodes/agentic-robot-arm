"""What one episode's policy produced. Shared by the baseline and the agent so the
harness scores both through exactly one code path."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeTrace:
    steps: list = field(default_factory=list)
    recoveries: int = 0
    escalations: int = 0
    claimed_success: bool = False
    claim_reason: str = ""
    stop_reason: str = ""
    l3_calls: int = 0
