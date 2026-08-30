import numpy as np
import pytest

from robotsim.cameras import OVERHEAD, OBLIQUE, Camera


@pytest.mark.parametrize("cam", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
@pytest.mark.parametrize("point", [
    (0.00, 0.00, 0.025),
    (0.10, -0.12, 0.025),
    (-0.08, 0.15, 0.025),
])
def test_project_unproject_roundtrip(cam: Camera, point):
    world = np.array(point)
    px, py = cam.project(world)
    assert 0 <= px < cam.width and 0 <= py < cam.height, "test point fell outside the frame"
    recovered = cam.unproject(px, py, z_plane=world[2])
    assert np.allclose(recovered, world, atol=3e-3), f"{recovered} != {world}"


def test_unproject_is_pure_geometry():
    """unproject must not need the simulator -- it is camera math, not a ground-truth lookup."""
    recovered = OVERHEAD.unproject(OVERHEAD.width // 2, OVERHEAD.height // 2, z_plane=0.0)
    assert recovered.shape == (3,)
    assert abs(recovered[2] - 0.0) < 1e-9
