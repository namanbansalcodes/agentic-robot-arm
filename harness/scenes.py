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


@dataclass(frozen=True)
class SuccessSpec:
    type: str                 # contained | contained_after_asking | honest_failure
    item: str
    container: str


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
                radius=o.get("radius", d["bowl_radius"]),
                height=o.get("height", d["bowl_height"]),
                size=tuple(o["size"]) if o.get("size") else None,
            ))
        scenes.append(SceneSpec(
            id=s["id"],
            failure_mode=s["failure_mode"],
            instruction=s["instruction"],
            objects=tuple(objects),
            success=SuccessSpec(**s["success"]),
            max_steps=s.get("max_steps", d["max_steps"]),
            max_retries_per_subtask=s.get("max_retries_per_subtask", d["max_retries_per_subtask"]),
            table=dict(d["table"]),
            human_answer=s.get("human_answer"),
        ))
    return scenes
