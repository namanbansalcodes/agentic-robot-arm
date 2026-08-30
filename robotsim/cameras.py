"""Camera model. Owns its view/projection matrices so that pixel -> world is our math.

Two views, deliberately:
  OVERHEAD  near-nadir, used for detection + unprojection (a nadir ray hits the
            table plane at a well-conditioned angle, so centroid error stays small)
  OBLIQUE   a human-legible three-quarter view, used for L3 visual verification and
            for the trajectory pages

This module never QUERIES simulator state -- it asks for no object pose, orientation,
or contact. `render()` does ask PyBullet to rasterize the live scene, which is exactly
what a real camera does: it returns pixels, not answers. That distinction is the whole
firewall. `project()` and `unproject()` are pure geometry and need no simulator at all.

The matrices are ours rather than panda-gym's because `sim.render()` builds its own
internally and hands back only pixels -- there is nothing to invert. Owning them is
what makes pixel -> world possible, and therefore what keeps coordinates out of the
VLM's hands.
"""
from dataclasses import dataclass

import numpy as np
import pybullet


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple            # camera position, world frame
    target: tuple         # look-at point, world frame
    up: tuple
    fov_deg: float
    width: int
    height: int
    near: float = 0.05
    far: float = 5.0

    # --- matrices -------------------------------------------------------
    @property
    def view_matrix(self) -> np.ndarray:
        m = pybullet.computeViewMatrix(list(self.eye), list(self.target), list(self.up))
        return np.array(m, dtype=np.float64).reshape(4, 4, order="F")

    @property
    def proj_matrix(self) -> np.ndarray:
        m = pybullet.computeProjectionMatrixFOV(
            fov=self.fov_deg, aspect=self.width / self.height, nearVal=self.near, farVal=self.far
        )
        return np.array(m, dtype=np.float64).reshape(4, 4, order="F")

    # --- geometry -------------------------------------------------------
    def project(self, world_xyz) -> tuple:
        """World point -> (pixel_x, pixel_y). Used by tests, never by the agent."""
        p = np.append(np.asarray(world_xyz, dtype=np.float64), 1.0)
        clip = self.proj_matrix @ (self.view_matrix @ p)
        ndc = clip[:3] / clip[3]
        px = (ndc[0] * 0.5 + 0.5) * self.width
        py = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height
        return float(px), float(py)

    def unproject(self, px: float, py: float, z_plane: float) -> np.ndarray:
        """(pixel, assumed height) -> world point, by intersecting the view ray with z=z_plane.

        This is the only place a 2D detection becomes a 3D target. The height is an
        assumption supplied by the caller (table height for cubes, rim height for
        bowls) -- not a ground-truth lookup. When the assumption is wrong the grasp
        misses, which is exactly the kind of honest failure the agent must catch.
        """
        inv = np.linalg.inv(self.proj_matrix @ self.view_matrix)
        ndc_x = (px / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - (py / self.height) * 2.0

        near_h = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_h = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near = near_h[:3] / near_h[3]
        far = far_h[:3] / far_h[3]

        direction = far - near
        if abs(direction[2]) < 1e-9:
            raise ValueError(f"{self.name}: view ray is parallel to the z={z_plane} plane")
        t = (z_plane - near[2]) / direction[2]
        return near + t * direction

    def render(self, physics_client_id: int) -> np.ndarray:
        """Rasterize with our matrices. Returns HxWx3 uint8 RGB."""
        _, _, rgba, _, _ = pybullet.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix.flatten(order="F").tolist(),
            projectionMatrix=self.proj_matrix.flatten(order="F").tolist(),
            renderer=pybullet.ER_TINY_RENDERER,
            physicsClientId=physics_client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]


OVERHEAD = Camera(
    name="overhead",
    eye=(0.0, 0.0, 0.85), target=(0.0, 0.0, 0.0), up=(1.0, 0.0, 0.0),
    fov_deg=45.0, width=480, height=480,
)

OBLIQUE = Camera(
    name="oblique",
    eye=(0.62, 0.52, 0.55), target=(0.0, 0.0, 0.03), up=(0.0, 0.0, 1.0),
    fov_deg=45.0, width=512, height=384,
)
