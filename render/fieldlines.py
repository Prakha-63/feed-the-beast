"""
Procedural dipole field lines + reference rings (Taichi kernel -> line list).

Geometry comes from physics.magnetosphere (axes, shells, r = L sin^2 theta);
this module only turns it into GPU line vertices.

Dipole field line of shell parameter L:   r(theta) = L sin^2(theta)
(theta = magnetic colatitude).  Lines are generated in the magnetic frame
(e1, e2, m) and mapped to scene radius with the log map.

Shells are spaced logarithmically between 1.05 R and the magnetospheric
boundary  r_max = min(r_m, r_lc):  the disk truncates the closed
magnetosphere at r_m, so a larger feeding rate visibly *compresses* the
field-line region.  This is the simplified dipole picture, not MHD.

Reference rings drawn in the equatorial plane:
    r_m   (orange)  magnetospheric radius
    r_co  (cyan)    corotation radius
    r_lc  (blue)    light-cylinder radius
"""
import math

import numpy as np
import taichi as ti

from render.scene_map import to_scene_ti, to_scene

N_AZ = 4          # azimuthal copies of each shell
RING_SEG = 96


@ti.data_oriented
class FieldLines:
    def __init__(self, n_shells, n_seg):
        self.n_shells = n_shells
        self.n_seg = n_seg
        self.n_lines = n_shells * N_AZ
        self.nv = self.n_lines * n_seg * 2
        self.verts = ti.Vector.field(3, ti.f32, self.nv)
        self.colors = ti.Vector.field(3, ti.f32, self.nv)
        self.ring_nv = 3 * RING_SEG * 2
        self.ring_verts = ti.Vector.field(3, ti.f32, self.ring_nv)
        self.ring_colors = ti.Vector.field(3, ti.f32, self.ring_nv)

    @ti.kernel
    def _build(self, e1: ti.types.vector(3, ti.f32), e2: ti.types.vector(3, ti.f32),
               m: ti.types.vector(3, ti.f32), R: ti.f32, r_max: ti.f32, tint: ti.f32):
        for li, k in ti.ndrange(self.n_lines, self.n_seg):
            shell = li // N_AZ
            az = li % N_AZ
            frac = (shell + 1.0) / self.n_shells
            L_min = 1.05 * R
            L = L_min * ti.pow(ti.max(r_max, L_min) / L_min, frac)   # log-spaced shells in [1.05 R, min(r_m, r_lc)]
            phi = 6.2831853 * az / N_AZ
            th0 = ti.asin(ti.sqrt(R / L))
            for j in ti.static(range(2)):
                t = (k + j) / self.n_seg
                th = th0 + (3.14159265 - 2.0 * th0) * t
                r = L * ti.sin(th) ** 2
                dirv = ti.sin(th) * ti.cos(phi) * e1 + ti.sin(th) * ti.sin(phi) * e2 + ti.cos(th) * m
                # no closed magnetosphere outside the star when the disk reaches the surface
                self.verts[(li * self.n_seg + k) * 2 + j] = dirv * to_scene_ti(r) if r_max > L_min else ti.Vector([0.0, 0.0, -1.0e5])
                w = 0.55 - 0.35 * frac
                self.colors[(li * self.n_seg + k) * 2 + j] = ti.Vector([0.25, 0.75, 0.75]) * w * tint

    @ti.kernel
    def _rings(self, s_m: ti.f32, s_co: ti.f32, s_lc: ti.f32, show_m: ti.f32):
        for ring, k in ti.ndrange(3, RING_SEG):
            s = s_m
            c = ti.Vector([1.0, 0.55, 0.15]) * show_m
            if ring == 1:
                s = s_co
                c = ti.Vector([0.2, 0.9, 0.9])
            if ring == 2:
                s = s_lc
                c = ti.Vector([0.35, 0.45, 1.0])
            for j in ti.static(range(2)):
                a = 6.2831853 * (k + j) / RING_SEG
                self.ring_verts[(ring * RING_SEG + k) * 2 + j] = ti.Vector([s * ti.cos(a), s * ti.sin(a), 0.0])
                self.ring_colors[(ring * RING_SEG + k) * 2 + j] = c

    def update(self, model):
        e1, e2, m = model.magnetosphere.basis        # axes from the rotation state
        tq = model.tq
        r_max = min(tq["r_m"], tq["r_lc"])
        self._build(e1.astype(np.float32), e2.astype(np.float32), m.astype(np.float32),
                    float(model.ns.radius), float(r_max), 0.6)
        s_m = to_scene(tq["r_m"]) if math.isfinite(tq["r_m"]) else 0.0
        self._rings(float(s_m), float(to_scene(tq["r_co"])), float(to_scene(tq["r_lc"])),
                    1.0 if math.isfinite(tq["r_m"]) else 0.0)
