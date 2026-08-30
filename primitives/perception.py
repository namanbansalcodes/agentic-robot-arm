"""Classical colour segmentation on rendered pixels. No simulator state, ever.

Cheap and imperfect on purpose: its mistakes are the failure modes the agent has to
notice and recover from. A perfect detector would make this whole project trivial
and the measurement meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Hue windows in OpenCV's 0-179 scale, plus saturation/value floors.
COLOR_RANGES = {
    "red":    [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    "green":  [((40, 90, 50), (85, 255, 255))],
    "blue":   [((100, 120, 60), (130, 255, 255))],
    "yellow": [((22, 120, 120), (34, 255, 255))],
}

MIN_AREA_PX = 120           # anything smaller is noise
# Measured on real overhead frames: a 5cm cube is 1,296-1,950 px, a bowl 13,389-14,845.
# From nadir the bowl reads as a FILLED square (its base plate is the same colour),
# so fill_ratio does NOT separate them -- both sit around 0.95. Area alone does,
# with a ~7x margin. fill_ratio is still recorded as evidence on the Detection.
BOWL_MIN_AREA_PX = 5000


@dataclass(frozen=True)
class Detection:
    id: str                 # e.g. "red_cube_1" -- the ONLY handle the VLM ever uses
    color: str
    kind: str               # cube | bowl
    centroid_px: tuple
    area_px: int
    bbox_px: tuple          # x, y, w, h
    fill_ratio: float
    where: str              # pixel-derived plain-English location


def _where(cx: float, cy: float, w: int, h: int) -> str:
    """Describe position IN THE PHOTO, not in world axes.

    The overhead camera uses up=(1,0,0), so the frame is rotated 90 degrees from
    world axes: image +x is world -y, image +y is world -x. Describing a pixel
    centroid as "left of the table" would therefore be actively wrong. The VLM is
    looking at this exact photo, so photo-relative language is both correct and the
    least confusing thing we can hand it. Anything needing real geometry goes
    through OVERHEAD.unproject, never through this string.
    """
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "centre")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    return f"{row}-{col} of the overhead photo"


def detect(rgb: np.ndarray) -> list[Detection]:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out: list[Detection] = []

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < MIN_AREA_PX:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            fill = area / float(max(bw * bh, 1))
            m = cv2.moments(c)
            cx = m["m10"] / m["m00"] if m["m00"] else x + bw / 2
            cy = m["m01"] / m["m00"] if m["m00"] else y + bh / 2
            kind = "bowl" if area >= BOWL_MIN_AREA_PX else "cube"
            blobs.append((cy, cx, area, (x, y, bw, bh), fill, kind))

        # Sort top-to-bottom, then left-to-right, so ids are stable run to run.
        blobs.sort(key=lambda b: (round(b[0], 1), round(b[1], 1)))
        counters: dict[str, int] = {}
        for cy, cx, area, bbox, fill, kind in blobs:
            counters[kind] = counters.get(kind, 0) + 1
            out.append(Detection(
                id=f"{color}_{kind}_{counters[kind]}",
                color=color, kind=kind,
                centroid_px=(float(cx), float(cy)), area_px=area, bbox_px=bbox,
                fill_ratio=float(fill), where=_where(cx, cy, w, h),
            ))
    return out
