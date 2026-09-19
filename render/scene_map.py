"""
Physical radius  ->  scene radius.

    s(r) = r / R_ns                       for r <= R_ns
    s(r) = 1 + K * ln(r / R_ns)           for r >  R_ns

The neutron star is 1 scene unit; the disk edge (3e5 km, 2.5e4 NS radii)
maps to ~23 units and the light cylinder (80 km) to ~5, so the magnetosphere
and the whole disk fit in one camera view.  Angles are preserved; radial
distances are NOT to scale.  Scene units are never used by the physics.
"""
import math

import taichi as ti

import config as cfg

R_NS = cfg.NS_RADIUS
K = cfg.SCENE_LOG_K
# GGUI draws a particle even with per-vertex radius 0, so hidden particles are
# parked here, far outside the camera's far plane.
HIDDEN = (0.0, 0.0, -1.0e5)


def to_scene(r):
    if r <= R_NS:
        return r / R_NS
    return 1.0 + K * math.log(r / R_NS)


def to_physical(s):
    if s <= 1.0:
        return s * R_NS
    return R_NS * math.exp((s - 1.0) / K)


@ti.func
def to_scene_ti(r: ti.f32) -> ti.f32:
    s = r / R_NS
    if r > R_NS:
        s = 1.0 + K * ti.log(r / R_NS)
    return s
