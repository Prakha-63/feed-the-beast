"""
Accretion disk: PHYSICAL STATE (this module) vs VISUAL PARTICLES (render/).

AccretionDisk  -- the simplified physical model used by the simulation --------

  A small Eulerian set of N_RINGS log-spaced Keplerian annuli between the
  neutron-star surface and a fixed outer edge DISK_OUTER_RADIUS.  Matter from
  the (unmodelled) companion enters at the outer edge at the user-set rate,
  carrying the local Keplerian specific angular momentum sqrt(G M r_out).
  Each ring i holds a mass m_i at fixed radius r_i.

    rotation        Omega_i = sqrt(G M / r_i^3)            (Keplerian)
    angular momentum J_i    = m_i sqrt(G M r_i)
    inflow          matter drains inward from ring i at    m_i / t_visc(r_i)
                    with the alpha-disk scaling  t_visc(r) ~ r^(3/2)
                    (constant alpha and H/r), normalised so the total
                    inflow time from r_circ to the star is DISK_VISCOUS_TIME.
    update          rings are advanced outer -> inner; each ring uses the
                    exact solution of  dm/dt = F_in - m/t_i  with the inflow
                    from the ring outside held constant over the step, so
                    the scheme is mass-conserving and stable for any dt
                    (inner rings have t_i ~ 0.1 s, outer ~ 1 day).
    inner edge      after the cascade the disk is truncated at the innermost
                    ring whose own inward flux satisfies  r_i >= r_m(flux_i)
                    (its ram pressure balances the dipole pressure there);
                    rings inside have no Keplerian disk and their content is
                    handed to the magnetosphere within the step (channelled
                    onto the star or propelled away, decided by the
                    neutron-star torque model).
    temperature     T_i = ( 3 G M Mdot_i / (8 pi sigma r_i^3) * (1 - sqrt(r_in/r_i)) )^(1/4)
                    with the *local* mass flux Mdot_i (Shakura & Sunyaev
                    1973), so inner and outer disk respond on their own
                    timescales.
    luminosity      L_disk = G M Mdot_mag / (2 r_in)   (half the binding energy
                    released down to the inner edge)

  Angular-momentum budget (all reported):
    Jdot_in    = Mdot_in * sqrt(G M r_out)        brought in at the outer edge
    Jdot_mag   = Mdot_mag * sqrt(G M r_m)        handed to the magnetosphere
                                                 (this is the term entering the
                                                 neutron-star accretion torque)
    Jdot_tidal = Jdot_in - Jdot_mag - dJ_disk/dt viscously transported to the
                                                 outer edge and out of the modelled
                                                 system (the binary is not modelled)

  NOT modelled: radial pressure, thermal/ionisation instability (the real
  J1023 transitions are probably instability driven), disk tilt/warp,
  irradiation, self-gravity, MHD.

"""
import math

import numpy as np

import config as cfg

SIGMA_SB = 5.670374419e-8
N_RINGS = 32


# ------------------------------------------------------------ ring model
class AccretionDisk:
    def __init__(self, r_in_min, r_out, n_rings=N_RINGS, total_inflow_time=cfg.DISK_VISCOUS_TIME):
        self.n = n_rings
        self.r_floor = r_in_min                 # neutron-star radius
        self.r_out = r_out                      # fixed outer edge (config.DISK_OUTER_RADIUS)
        self.total_inflow_time = total_inflow_time
        self.edges = np.geomspace(r_in_min, r_out, n_rings + 1)
        self.r = np.sqrt(self.edges[:-1] * self.edges[1:])       # geometric ring centres
        # alpha-disk scaling t ~ r^1.5, normalised so sum(t_i) = total inflow time
        w = self.r ** 1.5
        self.t_visc_ring = total_inflow_time * w / w.sum()
        self.m = np.zeros(n_rings)              # kg per ring
        self.flux = np.zeros(n_rings)           # kg/s inward mass flux leaving each ring (last step)
        self.r_in = r_out                       # current inner edge (= r_m clamped to the grid)
        self.mdot_mag = 0.0                     # kg/s handed to the magnetosphere (last step)
        self.mdot_in = 0.0                      # kg/s arriving from the stream (last step)
        self.jdot_in = self.jdot_mag = self.jdot_tidal = 0.0
        self.J_prev = 0.0

    # --------------------------------------------------------------- state
    @property
    def mass(self):
        return float(self.m.sum())

    @property
    def t_visc(self):
        """Total inflow time (for compatibility with the single-zone model)."""
        return self.total_inflow_time

    def omega(self, M):
        return np.sqrt(cfg.G * M / self.r ** 3)

    def angular_momentum(self, M):
        return float(np.sum(self.m * np.sqrt(cfg.G * M * self.r)))

    def active(self):
        """Rings that hold a Keplerian disk (outside the magnetosphere)."""
        return self.r >= self.r_in

    def temperature(self, M):
        """Shakura-Sunyaev effective temperature per ring from the local flux."""
        T = np.zeros(self.n)
        act = self.active() & (self.flux > 0)
        if act.any():
            fac = np.clip(1.0 - np.sqrt(self.r_in / self.r[act]), 0.0, None)
            T[act] = (3.0 * cfg.G * M * self.flux[act] / (8.0 * math.pi * SIGMA_SB * self.r[act] ** 3) * fac) ** 0.25
        return T

    def luminosity(self, M):
        return cfg.G * M * self.mdot_mag / (2.0 * self.r_in) if self.mdot_mag > 0 else 0.0

    def fill_fraction(self, reference_rate):
        """Per-ring mass relative to the steady state at reference_rate (visual density driver)."""
        steady = reference_rate * self.t_visc_ring
        return np.clip(self.m / steady, 0.0, 1.0)

    # --------------------------------------------------------------- step
    def step(self, mdot_in, dt, M, r_m_of_mdot):
        """Advance by dt.
        mdot_in        : mass supply rate at the outer edge (kg/s)
        M              : neutron-star mass
        r_m_of_mdot    : callable Mdot -> magnetospheric radius (neutron-star model)
        Returns the mean rate handed to the magnetosphere (kg/s).

        Truncation is self-consistent with the disk's own state: after the
        viscous cascade, the inner edge is the innermost ring whose local
        inward flux is large enough that  r_i >= r_m(flux_i)  (the ring's ram
        pressure holds the magnetosphere off).  Rings inside are handed over."""
        if not math.isfinite(dt) or dt <= 0.0:
            return self.mdot_mag
        self.mdot_in = mdot_in
        J0 = self.angular_momentum(M)
        # 1. viscous cascade over all rings, outer -> inner (exact per ring)
        F_in = mdot_in
        for i in range(self.n - 1, -1, -1):
            t_i = self.t_visc_ring[i]
            m0 = self.m[i]
            steady = F_in * t_i
            m1 = steady + (m0 - steady) * math.exp(-dt / t_i)
            out = max(F_in * dt - (m1 - m0), 0.0)     # exact mass leaving the ring inward
            self.m[i] = m1
            self.flux[i] = out / dt
            F_in = out / dt
        # 2. magnetospheric truncation from the local fluxes
        i_star = self.n                                 # default: no ring can resist -> all swept
        for i in range(self.n):
            if self.r[i] >= r_m_of_mdot(self.flux[i]):
                i_star = i
                break
        # mass leaving the cascade = outflow of the innermost ring (F_in after the
        # loop) + whatever the cascade left inside the truncation radius
        delivered = F_in * dt
        if i_star < self.n:
            self.r_in = max(self.r_floor, min(self.edges[i_star], r_m_of_mdot(self.flux[i_star])))
            delivered += float(self.m[:i_star].sum())
            self.flux[:i_star] = self.flux[i_star]      # matter inside free-falls at the inner-edge rate
            self.m[:i_star] = 0.0
        else:
            self.r_in = self.r_out
            delivered += float(self.m.sum())
            self.flux[:] = mdot_in
            self.m[:] = 0.0
        self.mdot_mag = delivered / dt
        # 3. angular-momentum budget
        J1 = self.angular_momentum(M)
        self.jdot_in = mdot_in * math.sqrt(cfg.G * M * self.r_out)     # Keplerian h at the outer edge
        self.jdot_mag = self.mdot_mag * math.sqrt(cfg.G * M * self.r_in)
        self.jdot_tidal = self.jdot_in - self.jdot_mag - (J1 - J0) / dt
        return self.mdot_mag

    def reset(self):
        self.m[:] = 0.0
        self.flux[:] = 0.0
        self.mdot_mag = self.mdot_in = 0.0
        self.jdot_in = self.jdot_mag = self.jdot_tidal = 0.0
        self.r_in = self.r_out

    def telemetry(self, M):
        T = self.temperature(M)
        act = self.active()
        return {
            "disk_mass_kg": self.mass,
            "disk_J": self.angular_momentum(M),
            "disk_r_in_km": self.r_in / 1e3,
            "disk_r_out_km": self.r_out / 1e3,
            "disk_T_max_K": float(T.max()) if act.any() else 0.0,
            "disk_T_outer_K": float(T[act][-1]) if act.any() else 0.0,
            "disk_L_W": self.luminosity(M),
            "disk_omega_in_rad_s": float(self.omega(M)[act][0]) if act.any() else 0.0,
            "disk_jdot_in": self.jdot_in, "disk_jdot_mag": self.jdot_mag, "disk_jdot_tidal": self.jdot_tidal,
            "disk_active_rings": int(act.sum()),
        }
