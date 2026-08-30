"""Shared scene fixtures for the test suite.

Two scene shapes the eval no longer ships but whose code paths still exist and still
have to work: an object the arm cannot reach, and a scene where the operator has
something to say. Both are built here by modifying a real scene rather than by adding
scenes to scenes.yaml that nothing measures -- scenes.yaml is the experiment, and a
scene in it that exists only to satisfy a test would show up in every report table.
"""
from __future__ import annotations

import dataclasses

from harness.scenes import SceneSpec, load_scenes

# Far outside the +/-0.22 workspace, and still inside the overhead camera's ~0.35 m
# half-width, so the object is DETECTED and then rejected as unreachable -- which is
# the path under test. Move it out of frame instead and the primitive would fail with
# "unknown_object", which is a different error entirely.
OUT_OF_REACH = (0.27, 0.20)


def scene_with_unreachable(base: SceneSpec, cube_name: str) -> SceneSpec:
    """`base` with one CUBE moved outside the arm's workspace.

    A cube, not a bowl, and that is load-bearing: a 0.075 m bowl at this position is
    clipped by the edge of the overhead frame, so its bounding box is truncated, its
    perceived centre is dragged back inside the workspace, and `place` cheerfully
    succeeds at a spot that is not where the bowl is. A cube is small enough to sit
    fully in frame, so it is detected correctly and then honestly rejected as out of
    reach -- which is the path these tests exist to exercise.
    """
    moved = [o for o in base.objects if o.name == cube_name]
    assert moved and moved[0].kind == "cube", f"{cube_name} is not a cube in {base.id}"
    objects = tuple(
        dataclasses.replace(o, position=OUT_OF_REACH) if o.name == cube_name else o
        for o in base.objects)
    return dataclasses.replace(base, id=f"{base.id}_unreachable", objects=objects)


def scene_with_human_answer(base: SceneSpec, answer: str) -> SceneSpec:
    return dataclasses.replace(base, human_answer=answer)


SCENES = {s.id: s for s in load_scenes()}
