"""
Accretion-disk visual particles (instanced, Taichi kernel).

Each particle lives at a *physical* radius r [m] and azimuth phi.  Per frame:
    phi += sqrt(G M / r^3) * kin_dt          Keplerian angular velocity
    r   -= (r / t_visc) * kin_dt             inward viscous drift (single-zone
                                             model: d ln r / dt = -1/t_visc)
    r < r_in  -> particle re-enters at r_out (steady recirculation)

Number of active particles per ring  = (N / n_rings) * m_k / m_k,steady(Mdot_max)
Colour = blackbody-ish ramp of the Shakura-Sunyaev temperature
    T(r) = ( 3 G M Mdot / (8 pi sigma r^3) )^(1/4)
Particle positions are mapped to scene units with the log map.
"""
import numpy as np
import taichi as ti

import config as cfg
from render.scene_map import to_scene_ti
from physics.disk import N_RINGS

SIGMA_SB = 5.670374419e-8


@ti.func
def temperature_color(T: ti.f32) -> ti.types.vector(3, ti.f32):
    """Piecewise-linear colour ramp in log10(T).  Brightness rises with T."""
    lt = ti.log(ti.max(T, 1.0)) / ti.log(10.0)
    c = ti.Vector([0.0, 0.0, 0.0])
    # anchors: (logT, colour)
    if lt < 3.3:
        c = ti.Vector([0.35, 0.05, 0.0]) * ti.max(0.0, (lt - 2.9) / 0.4)
    elif lt < 3.7:
        t = (lt - 3.3) / 0.4
        c = ti.Vector([0.35, 0.05, 0.0]) * (1 - t) + ti.Vector([1.0, 0.45, 0.12]) * t
    elif lt < 4.1:
        t = (lt - 3.7) / 0.4
        c = ti.Vector([1.0, 0.45, 0.12]) * (1 - t) + ti.Vector([1.0, 0.92, 0.75]) * t
    elif lt < 4.8:
        t = (lt - 4.1) / 0.7
        c = ti.Vector([1.0, 0.92, 0.75]) * (1 - t) + ti.Vector([0.75, 0.85, 1.0]) * t
    else:
        t = ti.min(1.0, (lt - 4.8) / 1.5)
        c = ti.Vector([0.75, 0.85, 1.0]) * (1 - t) + ti.Vector([0.9, 0.95, 1.0]) * t
    return c


@ti.data_oriented
class DiskParticles:
    """Visual particles grouped by physical ring: N/n_rings particles per ring.
    The number shown in ring k is  (N/n_rings) * m_k / m_k,steady(Mdot_max),
    their colour is the ring's Shakura-Sunyaev temperature, their azimuth
    advances at the Keplerian rate of their own radius and they drift inward at
    the ring's viscous rate r / t_visc(r), recirculating within the ring band."""

    def __init__(self, n, rng, n_rings=N_RINGS):
        self.n_rings = n_rings
        self.per_ring = max(1, n // n_rings)
        self.n = self.per_ring * n_rings
        self.r = ti.field(ti.f32, self.n)
        self.phi = ti.field(ti.f32, self.n)
        self.zoff = ti.field(ti.f32, self.n)
        self.pos = ti.Vector.field(3, ti.f32, self.n)
        self.color = ti.Vector.field(3, ti.f32, self.n)
        self.radius = ti.field(ti.f32, self.n)
        self.ring_lo = ti.field(ti.f32, n_rings)
        self.ring_hi = ti.field(ti.f32, n_rings)
        self.ring_t = ti.field(ti.f32, n_rings)
        self.ring_state = ti.Vector.field(2, ti.f32, n_rings)     # (fill fraction, temperature) per ring
        self.rng = rng
        self._grid_key = None
        self.phi.from_numpy((rng.random(self.n) * 2 * np.pi).astype(np.float32))
        self.zoff.from_numpy((rng.standard_normal(self.n) * 0.5).astype(np.float32))
        self._u = rng.random(self.n).astype(np.float32)    # deterministic radial placement within the band

    def _bind_grid(self, disk):
        key = (float(disk.edges[0]), float(disk.edges[-1]), disk.n)
        if key == self._grid_key:
            return
        self._grid_key = key
        self.ring_lo.from_numpy(disk.edges[:-1].astype(np.float32))
        self.ring_hi.from_numpy(disk.edges[1:].astype(np.float32))
        self.ring_t.from_numpy(disk.t_visc_ring.astype(np.float32))
        k = np.arange(self.n) // self.per_ring
        r = disk.edges[k] * (disk.edges[k + 1] / disk.edges[k]) ** self._u
        self.r.from_numpy(r.astype(np.float32))

    @ti.kernel
    def _step(self, dt: ti.f32, GM: ti.f32, r_in: ti.f32, per_ring: ti.i32, prad: ti.f32):
        for i in range(self.n):
            k = i // per_ring
            j = i - k * per_ring
            r = self.r[i]
            omega = ti.sqrt(GM / (r * r * r))
            phi = self.phi[i] + omega * dt                       # Keplerian rotation of this radius
            r = r - r / self.ring_t[k] * dt                      # viscous inflow of this ring
            lo, hi = self.ring_lo[k], self.ring_hi[k]
            if r < lo:                                           # recirculate inside the ring band
                r = hi - (lo - r) % (hi - lo)
            self.r[i] = r
            self.phi[i] = phi - 6.2831853 * ti.floor(phi / 6.2831853)
            s = to_scene_ti(r)
            h = 0.012 * s                                        # visual thickness only (H/r not modelled)
            self.pos[i] = ti.Vector([s * ti.cos(phi), s * ti.sin(phi), self.zoff[i] * h])
            self.color[i] = temperature_color(self.ring_state[k][1])
            alive = (j < per_ring * self.ring_state[k][0]) and (r >= r_in)
            # small sprites: 0.045 near the star (dense hot inner rings) -> ~0.02 at the outer edge
            self.radius[i] = prad * ti.max(0.45, 1.0 - 0.03 * s) if alive else 0.0
            if not alive:
                self.pos[i] = ti.Vector([0.0, 0.0, -1.0e5])

    def update(self, model):
        self._bind_grid(model.disk)
        self.ring_state.from_numpy(np.stack([model.disk_ring_fill, model.disk_ring_T], axis=1).astype(np.float32))
        self._step(float(model.kin_dt), float(cfg.G * model.ns.mass), float(model.disk.r_in), self.per_ring, 0.045)
