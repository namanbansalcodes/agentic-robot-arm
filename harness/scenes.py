"""Loads scenes.yaml into typed specs. Pure data -- no simulator, no VLM."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml

SCENES_PATH = pathlib.Path(__file__).resolve().parent.parent / "scenes.yaml"


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str                 # cube | bowl | wall
    color: str
    position: tuple           # (x, y) on the table; z is derived from kind
    half_extent: float = 0.025
    mass: float = 0.08
    lateral_friction: float = 2.0
    radius: float = 0.075
    height: float = 0.05
    size: Optional[tuple] = None      # walls only
    # Spawn this cube INSIDE the named bowl instead of on the open table. The bowl
    # must be declared earlier in the scene's object list. Used by the memory scenes,
    # where the starting state is half of the puzzle: mem_swap starts both blocks in
    # the wrong bowls, mem_recall starts one block already in the right one. Spawning
    # them by coordinate and hoping the per-seed jitter kept them inside would make
    # the PRECONDITION of those scenes a coin flip; naming the bowl makes it exact.
    in_bowl: Optional[str] = None


@dataclass(frozen=True)
class SuccessSpec:
    """ONE predicate for both task families.

    "put every block in the blue bowl" and "put every block in the bowl of its
    matching colour" are the same thing written twice: a list of (item, container)
    pairs, all of which must hold. Keeping two success types would have meant two
    scoring paths, two ways to be subtly wrong, and a metric that could not compare
    the two families on one axis.
    """

    type: str                            # pairs | ordered_pairs | recall
    pairs: tuple = ()                    # ((item, container), ...), ALL must hold
    # `recall` only: pairs that must NOT hold at the end. The block that started in
    # the bowl has to come back out, and "not in the bowl" is a graded condition in
    # its own right rather than an afterthought -- it counts toward progress like any
    # other, because taking it out is half the job.
    excluded: tuple = ()


@dataclass(frozen=True)
class DisturbanceSpec:
    """An adversarial intervention the HARNESS performs, never the agent.

    After `after_placements` pairs are satisfied, one already-placed block is moved
    back onto the table at `to`. Nothing agent-side is told this happened -- noticing
    it requires looking, which is the whole point of the cell.
    """

    after_placements: int
    action: str                          # eject
    to: tuple                            # (x, y) landing spot, inside the workspace


@dataclass(frozen=True)
class SceneSpec:
    id: str
    failure_mode: str
    instruction: str
    objects: tuple
    success: SuccessSpec
    max_steps: int
    max_retries_per_subtask: int
    table: dict
    human_answer: Optional[str] = None
    disturbance: Optional[DisturbanceSpec] = None
    # `recall` only: which block was already in the bowl at spawn. This is the JUDGE's
    # record of the answer. It is never shown to the agent -- noticing it in the first
    # photo and remembering it once the bowl fills up is the whole task.
    initially_in_bowl: Optional[str] = None


def load_scenes(path: pathlib.Path = SCENES_PATH) -> list[SceneSpec]:
    raw = yaml.safe_load(path.read_text())
    d = raw["defaults"]
    scenes = []
    for s in raw["scenes"]:
        objects = []
        for o in s["objects"]:
            objects.append(ObjectSpec(
                name=o["name"],
                kind=o["kind"],
                color=o["color"],
                position=tuple(o["position"]),
                half_extent=o.get("half_extent", d["cube_half_extent"]),
                mass=o.get("mass", d["cube_mass"]),
                lateral_friction=o.get("lateral_friction", d["lateral_friction"]),
                # `bowl_radius` is the scene-file spelling (it reads as the default it
                # overrides); `radius` stays accepted so older scene files still load.
                radius=o.get("bowl_radius", o.get("radius", d["bowl_radius"])),
                height=o.get("height", d["bowl_height"]),
                size=tuple(o["size"]) if o.get("size") else None,
                in_bowl=o.get("in_bowl"),
            ))
        success = s["success"]
        disturbance = s.get("disturbance")
        scenes.append(SceneSpec(
            id=s["id"],
            failure_mode=s["failure_mode"],
            instruction=s["instruction"],
            objects=tuple(objects),
            success=SuccessSpec(
                type=success["type"],
                pairs=tuple(tuple(pair) for pair in success.get("pairs", ())),
                excluded=tuple(tuple(pair) for pair in success.get("excluded", ())),
            ),
            max_steps=s.get("max_steps", d["max_steps"]),
            max_retries_per_subtask=s.get("max_retries_per_subtask", d["max_retries_per_subtask"]),
            table=dict(d["table"]),
            human_answer=s.get("human_answer"),
            initially_in_bowl=s.get("initially_in_bowl"),
            disturbance=DisturbanceSpec(
                after_placements=int(disturbance["after_placements"]),
                action=str(disturbance["action"]),
                to=tuple(disturbance["to"]),
            ) if disturbance else None,
        ))
    return scenes
