"""What every primitive hands back.

One shape for success and failure alike. Every verification layer in this project --
the L1 error string, the L2 aperture check, the L3 visual re-look -- reads its evidence
out of this object, so it is deliberately generous: it carries the outcome, the
proprioception, the fresh detections, and the path to the image they came from.

`to_model_text()` is the narrow end: the compact form injected into the next planning
prompt. It reports the robot's OWN pose (proprioception, which the agent is allowed to
know) and names objects by id and photo-relative location -- never by world coordinate.
Widening it to include object positions would hand the VLM the coordinates the whole
firewall exists to withhold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Feedback:
    """What a primitive hands back. Generous on purpose -- this is the raw material
    every verification layer runs on."""
    primitive: str
    args: dict
    status: str                       # ok | error
    error: Optional[str] = None       # the L1 signal, e.g. "unreachable: ..."
    fingers_width: float = 0.0
    ee_position: tuple = (0.0, 0.0, 0.0)
    detections: list = field(default_factory=list)   # list[dict], pixel-derived
    image_path: Optional[str] = None
    sim_steps: int = 0
    note: Optional[str] = None

    def to_model_text(self) -> str:
        """The compact form injected into the next planning prompt."""
        lines = [f"{self.primitive}({self.args}) -> {self.status}"]
        if self.error:
            lines.append(f"  error: {self.error}")
        lines.append(f"  gripper_aperture_m: {self.fingers_width:.4f}")
        lines.append(f"  ee_position_m: [{', '.join(f'{v:.3f}' for v in self.ee_position)}]")
        if self.detections:
            seen = ", ".join(f"{d['id']} ({d['where']})" for d in self.detections)
            lines.append(f"  visible: {seen}")
        else:
            lines.append("  visible: nothing detected")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)
