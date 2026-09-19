"""
Simplified particle outflow model (physical drivers; visuals in render/wind.py).

Two channels, both computed from the neutron-star / disk state every step:

POLAR OUTFLOW  (from the magnetic-pole regions, along the magnetic axis)
    launch region   the dipole polar cap: open field lines (those reaching
                    beyond the light cylinder) have surface footpoints within
                        theta_cap = asin( sqrt( R / r_lc ) )
                    of each magnetic pole (r_lc = c / Omega).  Outflow particles
                    start on the cap and leave radially, so the cone half-angle
                    IS the polar-cap angle - a faster spin gives a wider cap.
    power           P_polar = L_sd + F_JET * L_acc
                    rotation-powered pulsar wind carries the spin-down power;
                    an assumed fraction F_JET of the accretion power leaves in
                    polar outflow when matter is channelled onto the caps.
    intensity       I_polar = activity.log_index(P_polar)  - the same
                    logarithmic activity scale as the global activity index
                    (physics/activity.py), so higher activity -> stronger
                    outflow, lower -> weaker.
    speed           V_POLAR = 0.3 c (assumed bulk speed of the wind)
    colour weight   rotation-powered fraction  L_sd / P_polar  (blue) vs
                    accretion-powered (orange)

EQUATORIAL OUTFLOW  (propeller ejection from the disk inner edge)
    rate            Mdot_ej = Mdot_disk - Mdot_ns   (from the torque model)
    intensity       I_eq = clip( (Mdot_ej / MAX_TRANSFER_RATE)^(1/3) )
                    (the cubic-root mapping used for all matter-flow counts)
    speed           v_esc(r_m) = sqrt( 2 G M / r_m )     (computed)

VISUAL PARTICLE COUNT  n_channel = N_channel * I_channel  (bounded by the
allocated Taichi fields; a few thousand particles at most).  Particle counts
are display proxies for the computed power / rate; the physics never reads
them back.  The only randomness is the uniform distribution of launch points
over the cap and of emission times, which does not change the rates.
"""
import math

import config as cfg

from physics.activity import log_index

F_JET = cfg.ACTIVITY_JET_FRACTION   # fraction of accretion power assumed to leave as polar outflow
V_POLAR = 0.3 * cfg.C    # m/s


def polar_cap_angle(R, r_lc):
    """Half-angle of the open-field-line footprint about the magnetic pole."""
    if not math.isfinite(r_lc) or r_lc <= R:
        return math.pi / 2 if r_lc <= R else 0.0
    return math.asin(math.sqrt(R / r_lc))


class Outflow:
    def __init__(self, ns):
        self.ns = ns
        self.polar_power = 0.0
        self.polar_index = 0.0
        self.polar_rotation_fraction = 1.0
        self.cap_angle = 0.0
        self.eq_mdot = 0.0
        self.eq_index = 0.0
        self.eq_speed = 0.0
        self.eq_radius = ns.radius

    def update(self, tq):
        ns = self.ns
        self.cap_angle = polar_cap_angle(ns.radius, tq["r_lc"])
        self.polar_power = tq["L_sd"] + F_JET * tq["L_acc"]
        self.polar_index = log_index(self.polar_power)
        self.polar_rotation_fraction = tq["L_sd"] / self.polar_power if self.polar_power > 0 else 1.0
        self.eq_mdot = max(tq["mdot_ejected"], 0.0)
        self.eq_index = min(1.0, (self.eq_mdot / cfg.MAX_TRANSFER_RATE) ** (1.0 / 3.0))
        self.eq_radius = tq["r_m"] if math.isfinite(tq["r_m"]) else ns.radius
        self.eq_speed = math.sqrt(2.0 * cfg.G * ns.mass / self.eq_radius)

    @property
    def polar_speed(self):
        return V_POLAR

    def visual_counts(self, n_polar_max, n_eq_max):
        return int(n_polar_max * self.polar_index), int(n_eq_max * self.eq_index)

    def telemetry(self):
        return {
            "outflow_polar_power_W": self.polar_power,
            "outflow_polar_index": self.polar_index,
            "outflow_polar_rotation_fraction": self.polar_rotation_fraction,
            "outflow_cap_angle_deg": math.degrees(self.cap_angle),
            "outflow_polar_speed_c": V_POLAR / cfg.C,
            "outflow_eq_mdot_msun_yr": self.eq_mdot * cfg.YEAR / cfg.M_SUN,
            "outflow_eq_index": self.eq_index,
            "outflow_eq_speed_km_s": self.eq_speed / 1e3,
        }
