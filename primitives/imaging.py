"""ndarray -> PNG bytes. Agent-safe: pixels in, pixels out."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()
