"""
Visual particle outflow (Taichi fields, bounded count) driven by physics/outflow.py.

    POLAR      n = N/2 * I_polar particles launched on the dipole polar caps
               (within the computed cap angle of +/- m), moving radially at
               V_POLAR; colour blends blue (rotation-powered fraction) and
               orange (accretion-powered fraction).
    EQUATORIAL n = N/2 * I_eq particles launched at the disk inner edge r_m in
               the disk plane, moving at the computed escape speed; magenta.

Particles advance in *physical* radius by v * kin_dt and are log-mapped to
scene units.  Because 0.3 c crosses the whole rendered region in far less
than one frame at any warp above slow-motion, the displayed radial advance per
frame is capped at WIND_MAX_STEP scene units (documented visual limit).  In
slow-motion (1 s = 1 ms) the cap is inactive.  Emission is staggered in time
(respawn probability ~ 9 / lifetime in frames, steady visible fraction ~0.9 of
the budget) so the stream is continuous; the number of visible particles is set
by the model's indices, never by chance.

Determinism: every random draw comes from a per-particle 32-bit LCG whose
state is seeded from the run seed (numpy), never from Taichi's global RNG, so
RESTART with the same seed replays the outflow bit-for-bit.
"""
import math

import numpy as np
import taichi as ti

import config as cfg
from render.scene_map import to_scene_ti, R_NS, K

WIND_MAX_STEP = 0.30       # scene units per frame
S_MAX = 24.0               # particles die beyond this scene radius
EQ_SPREAD = 0.35           # rad vertical spread of the equatorial ejection


@ti.data_oriented
class Wind:
    def __init__(self, n, rng):
        self.n = n
        self.half = n // 2
        self.r = ti.field(ti.f32, n)             # physical radius, m
        self.dirv = ti.Vector.field(3, ti.f32, n)
        self.alive = ti.field(ti.i32, n)
        self.pos = ti.Vector.field(3, ti.f32, n)
        self.color = ti.Vector.field(3, ti.f32, n)
        self.radius = ti.field(ti.f32, n)
        self.rng_state = ti.field(ti.u32, n)
        self.rng_state.from_numpy(rng.integers(1, 2 ** 32 - 1, n, dtype=np.uint64).astype(np.uint32))
        # initial population spread along the flow (no synchronous burst): polar
        # half within alpha + cap(r_lc at the observed spin) of +/- z (the
        # time-averaged hollow cone of a sweeping axis), equatorial half in the plane
        s0 = rng.uniform(2.0, S_MAX, n)
        self.r.from_numpy((R_NS * np.exp((s0 - 1.0) / K)).astype(np.float32))
        d = np.empty((n, 3))
        half = self.half
        cone0 = cfg.MAGNETIC_MISALIGNMENT + math.asin(math.sqrt(cfg.NS_RADIUS * 2 * math.pi / (cfg.C * cfg.NS_SPIN_PERIOD_OBS)))
        a = cone0 * np.sqrt(rng.random(half))
        b = 2 * np.pi * rng.random(half)
        sgn = np.where(rng.random(half) < 0.5, -1.0, 1.0)
        d[:half] = np.stack([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), sgn * np.cos(a)], axis=1)
        ph = 2 * np.pi * rng.random(n - half)
        el = EQ_SPREAD * rng.standard_normal(n - half)
        d[half:] = np.stack([np.cos(el) * np.cos(ph), np.cos(el) * np.sin(ph), np.sin(el)], axis=1)
        self.dirv.from_numpy(d.astype(np.float32))
        self.alive.fill(1)

    @ti.func
    def _rand(self, i) -> ti.f32:
        """Uniform [0,1) from particle i's own LCG (Numerical Recipes constants)."""
        self.rng_state[i] = self.rng_state[i] * ti.u32(1664525) + ti.u32(1013904223)
        return ti.f32(self.rng_state[i] >> ti.u32(8)) * (1.0 / 16777216.0)

    @ti.func
    def _randn(self, i) -> ti.f32:
        """Approximate standard normal: sum of 4 uniforms, variance-matched."""
        return (self._rand(i) + self._rand(i) + self._rand(i) + self._rand(i) - 2.0) * 1.7320508

    @ti.kernel
    def _step(self, dt: ti.f32, n_polar: ti.i32, n_eq: ti.i32, m: ti.types.vector(3, ti.f32),
              e1: ti.types.vector(3, ti.f32), e2: ti.types.vector(3, ti.f32),
              R: ti.f32, r_eq: ti.f32, v_polar: ti.f32, v_eq: ti.f32, cap: ti.f32,
              rot_frac: ti.f32, pol_int: ti.f32, eq_int: ti.f32, spawn_p: ti.f32):
        c_polar = ti.Vector([0.55, 0.75, 1.0]) * rot_frac + ti.Vector([1.0, 0.6, 0.3]) * (1.0 - rot_frac)
        for i in range(self.n):
            polar = i < self.half
            budget = (i < n_polar) if polar else ((i - self.half) < n_eq)
            if self.alive[i] == 0 and budget and self._rand(i) < spawn_p:   # staggered emission
                self.alive[i] = 1
                if polar:
                    # launch point uniformly over the polar cap (area-uniform in the cone)
                    sgn = 1.0 if self._rand(i) < 0.5 else -1.0
                    a = cap * ti.sqrt(self._rand(i))
                    b = 6.2831853 * self._rand(i)
                    self.dirv[i] = sgn * ti.cos(a) * m + ti.sin(a) * (ti.cos(b) * e1 + ti.sin(b) * e2)
                    self.r[i] = R * (1.05 + 0.6 * self._rand(i))     # spread launch radius: no synchronous rings
                else:
                    ph = 6.2831853 * self._rand(i)
                    el = EQ_SPREAD * self._randn(i)
                    self.dirv[i] = ti.Vector([ti.cos(el) * ti.cos(ph), ti.cos(el) * ti.sin(ph), ti.sin(el)])
                    self.r[i] = r_eq * (1.05 + 0.6 * self._rand(i))
                if dt > 60.0:   # fast warp: stagger along the flow instead of bursting
                    self.r[i] = self.r[i] * ti.exp(self._rand(i) * 6.0)
            if self.alive[i] == 1 and budget:
                v = v_polar if polar else v_eq
                s_old = to_scene_ti(self.r[i])
                r_new = self.r[i] + v * dt
                s_new = to_scene_ti(r_new)
                if s_new - s_old > WIND_MAX_STEP:
                    s_new = s_old + WIND_MAX_STEP
                    r_new = R * ti.exp((s_new - 1.0) / K)
                self.r[i] = r_new
                self.pos[i] = self.dirv[i] * s_new
                fade = ti.max(0.0, 1.0 - (s_new - 2.0) / (S_MAX - 2.0)) ** 1.2
                if polar:
                    self.color[i] = c_polar * fade * (0.45 + 0.55 * pol_int)
                    self.radius[i] = 0.02 + 0.03 * s_new / S_MAX
                else:
                    self.color[i] = ti.Vector([1.0, 0.35, 0.75]) * fade * (0.4 + 0.6 * eq_int)
                    self.radius[i] = 0.03 + 0.03 * s_new / S_MAX
                if s_new >= S_MAX:
                    self.alive[i] = 0
            if self.alive[i] == 0 or not budget:
                self.radius[i] = 0.0
                self.pos[i] = ti.Vector([0.0, 0.0, -1.0e5])
                if not budget:
                    self.alive[i] = 0

    def update(self, model, basis):
        e1, e2, m = basis
        of = model.outflow
        n_polar, n_eq = of.visual_counts(self.half, self.n - self.half)
        dt = float(model.kin_dt)
        # respawn probability ~ 9 / lifetime in frames: a freed slot is re-emitted
        # within ~1/9 of a lifetime, so the visible count stays ~ the budget
        # (steady fraction 0.9) while emission remains spread in time
        step_scene = min(WIND_MAX_STEP, K * math.log1p(of.polar_speed * dt / model.ns.radius)) if dt > 0 else 0.0
        spawn_p = min(1.0, max(0.05, 9.0 * step_scene / (S_MAX - 2.0))) if dt > 0 else 0.0
        self._step(dt, n_polar, n_eq, m.astype(np.float32), e1.astype(np.float32), e2.astype(np.float32),
                   float(model.ns.radius), float(of.eq_radius), float(of.polar_speed), float(of.eq_speed),
                   float(of.cap_angle), float(of.polar_rotation_fraction), float(of.polar_index),
                   float(of.eq_index), float(spawn_p))

    def visible_counts(self):
        """(polar, equatorial) particles currently shown - for validation."""
        rad = self.radius.to_numpy()
        return int((rad[:self.half] > 0).sum()), int((rad[self.half:] > 0).sum())
