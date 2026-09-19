"""
Pulsar / radiation beams: two opposed cones along the magnetic axis.

Lightweight procedural geometry (no volumetrics):
    * wire-frame cone per beam: N_GEN generator lines from the stellar surface
      to BEAM_LENGTH plus N_RINGS cross-section rings, half-angle
      BEAM_HALF_ANGLE (the same parameter the pulse-profile model uses)
    * a bright core line along the axis
    * a sparse set of small particles inside the cone for depth
All of it is re-oriented every frame with the magnetic basis derived from
rotation axis + misalignment + spin phase (Omega = J/I): the beams sweep
because the phase advances; there is no beam rotation rate of its own.

Colour carries information (physics/radiation.py):
    blue   ~ I_rot  rotation-powered (L_sd), quenched when a disk is present
    orange ~ I_acc  accretion-powered X-ray hot spots (L_acc)
Brightness and particle size scale with the total intensity; below 2 %
nothing is drawn.
"""
import numpy as np
import taichi as ti

from physics.radiation import BEAM_HALF_ANGLE
from render.scene_map import to_scene

# Beam length is not fixed: cones are drawn out to the light cylinder r_lc = c/Omega
# (scene radius from the log map), so a faster spin gives a shorter, tighter beam region.
N_GEN = 12                  # generator lines per cone
N_RINGS = 2                 # cross-section rings per cone
RING_SEG = 36
CORE_SEGMENTS = 16


@ti.data_oriented
class Beams:
    def __init__(self, n, rng):
        # sparse fill particles
        self.n = n
        self.t = ti.field(ti.f32, n)
        self.rho = ti.field(ti.f32, n)
        self.psi = ti.field(ti.f32, n)
        self.sign = ti.field(ti.f32, n)
        self.pos = ti.Vector.field(3, ti.f32, n)
        self.color = ti.Vector.field(3, ti.f32, n)
        self.radius = ti.field(ti.f32, n)
        self.t.from_numpy((rng.random(n) ** 1.4).astype(np.float32))
        self.rho.from_numpy((rng.random(n) ** 0.5).astype(np.float32))
        self.psi.from_numpy((rng.random(n) * 2 * np.pi).astype(np.float32))
        self.sign.from_numpy(np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32))
        # wire-frame: per beam  N_GEN generators + N_RINGS rings + core
        self.n_gen_v = 2 * N_GEN * 2
        self.n_ring_v = 2 * N_RINGS * RING_SEG * 2
        self.n_core_v = 2 * CORE_SEGMENTS * 2
        self.wire_nv = self.n_gen_v + self.n_ring_v
        self.wire_verts = ti.Vector.field(3, ti.f32, self.wire_nv)
        self.wire_colors = ti.Vector.field(3, ti.f32, self.wire_nv)
        self.core_verts = ti.Vector.field(3, ti.f32, self.n_core_v)
        self.core_colors = ti.Vector.field(3, ti.f32, self.n_core_v)

    @ti.kernel
    def _build(self, e1: ti.types.vector(3, ti.f32), e2: ti.types.vector(3, ti.f32),
               m: ti.types.vector(3, ti.f32), blue: ti.f32, orange: ti.f32, BEAM_LENGTH: ti.f32):
        inten = ti.min(1.0, blue + orange)
        on = inten > 0.02
        hidden = ti.Vector([0.0, 0.0, -1.0e5])
        c_beam = ti.Vector([0.45, 0.65, 1.0]) * blue + ti.Vector([1.0, 0.55, 0.2]) * orange
        tan_w = ti.tan(BEAM_HALF_ANGLE)
        # --- fill particles ---------------------------------------------------
        for i in range(self.n):
            t = self.t[i]
            d = 1.0 + t * BEAM_LENGTH
            rr = self.rho[i] * d * tan_w
            p = self.sign[i] * d * m + rr * (ti.cos(self.psi[i]) * e1 + ti.sin(self.psi[i]) * e2)
            fade = (1.0 - t) ** 1.2
            self.color[i] = c_beam * (0.35 + 0.65 * fade)
            if on:
                self.pos[i] = p
                self.radius[i] = (0.015 + 0.03 * t) * ti.min(1.0, 0.4 + inten)
            else:
                self.pos[i] = hidden
                self.radius[i] = 0.0
        # --- generators -------------------------------------------------------
        for k in range(2 * N_GEN):
            sgn = 1.0 if k < N_GEN else -1.0
            g = k if k < N_GEN else k - N_GEN
            psi = 6.2831853 * g / N_GEN
            u = ti.cos(psi) * e1 + ti.sin(psi) * e2
            d1 = 1.0 + BEAM_LENGTH
            p0 = sgn * m + tan_w * u
            p1 = sgn * d1 * m + d1 * tan_w * u
            self.wire_verts[2 * k] = p0 if on else hidden
            self.wire_verts[2 * k + 1] = p1 if on else hidden
            self.wire_colors[2 * k] = c_beam * (0.9 * inten + 0.1)
            self.wire_colors[2 * k + 1] = c_beam * 0.08
        # --- rings ------------------------------------------------------------
        for k in range(2 * N_RINGS * RING_SEG):
            beam = k // (N_RINGS * RING_SEG)
            rem = k - beam * N_RINGS * RING_SEG
            ring = rem // RING_SEG
            seg = rem - ring * RING_SEG
            sgn = 1.0 if beam == 0 else -1.0
            frac = (ring + 1.0) / N_RINGS
            d = 1.0 + frac * BEAM_LENGTH
            base = self.n_gen_v + 2 * k
            for j in ti.static(range(2)):
                a = 6.2831853 * (seg + j) / RING_SEG
                u = ti.cos(a) * e1 + ti.sin(a) * e2
                p = sgn * d * m + d * tan_w * u
                self.wire_verts[base + j] = p if on else hidden
                self.wire_colors[base + j] = c_beam * ((1.0 - frac) * 0.6 + 0.08) * (0.5 + 0.5 * inten)
        # --- core -------------------------------------------------------------
        for k in range(2 * CORE_SEGMENTS):
            sgn = 1.0 if k < CORE_SEGMENTS else -1.0
            j = k if k < CORE_SEGMENTS else k - CORE_SEGMENTS
            t0 = j / CORE_SEGMENTS
            t1 = (j + 1) / CORE_SEGMENTS
            self.core_verts[2 * k] = sgn * (1.0 + t0 * BEAM_LENGTH) * m if on else hidden
            self.core_verts[2 * k + 1] = sgn * (1.0 + t1 * BEAM_LENGTH) * m if on else hidden
            w = (1.0 - t0) ** 1.5 * inten
            self.core_colors[2 * k] = (c_beam * 0.6 + ti.Vector([0.4, 0.4, 0.4])) * w
            self.core_colors[2 * k + 1] = (c_beam * 0.6 + ti.Vector([0.4, 0.4, 0.4])) * w

    def update(self, model, basis):
        e1, e2, m = basis
        length = max(1.0, to_scene(model.tq["r_lc"]) - 1.0)      # star surface -> light cylinder
        self._build(e1.astype(np.float32), e2.astype(np.float32), m.astype(np.float32),
                    float(model.beam_blue), float(model.beam_orange), float(length))
