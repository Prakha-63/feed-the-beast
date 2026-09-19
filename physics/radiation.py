"""
Pulsar radiation: two opposed beams along the magnetic axis.

GEOMETRY   beam directions  +m and -m,  m = magnetic_axis(spin_phase, alpha)
           (physics/magnetosphere.py).  The spin phase is advanced by the
           neutron-star model with Omega = J/I, so the beams sweep at exactly
           the computed rotation rate - there is no beam rotation speed of its
           own.  Beam half-opening angle BEAM_HALF_ANGLE (model parameter).

INTENSITY  two emission channels from the modelled activity (physics/activity.py):
    rotation-powered  I_rot = rotation_intensity(L_sd, state)   quenched when a
                      disk is present (observed-inspired; simplified factor)
    accretion-powered I_acc = accretion_intensity(L_acc)        X-ray hot spots
                      at the magnetic poles, carried by the same two beams
    I_total = I_rot + I_acc  (clipped to 1 for display)

PULSES     an observer at inclination i (angle to the spin axis, azimuth 0)
           sees a pulse whenever a beam axis passes within the beam width of
           the line of sight:
    I(phi) = I_total * sum_{+/-} exp( -(angle(+/-m(phi), n) / w)^2 )
    pulse period    = P = 2 pi / Omega    (one main pulse per rotation)
    pulse frequency = f = 1 / P
    interpulse      from the -m beam if |pi - alpha - i| < w  (then two per P)
    next pulse      when the m azimuth reaches the observer azimuth:
                    t_next = ((-spin_phase) mod 2 pi) / Omega
Everything here is derived; nothing is animated independently.
"""
import math

import numpy as np

import config as cfg
from physics import states
from physics.magnetosphere import magnetic_axis
from physics.activity import rotation_intensity, accretion_intensity

BEAM_HALF_ANGLE = math.radians(12.0)        # model parameter
OBSERVER_INCLINATION = math.radians(40.0)   # assumed line of sight vs spin axis (5 deg from the 35 deg beam axis)


def observer_direction(inclination=OBSERVER_INCLINATION):
    return np.array([math.sin(inclination), 0.0, math.cos(inclination)])


def beam_pattern(cos_angle, half_angle=BEAM_HALF_ANGLE):
    """Gaussian beam: relative intensity vs cosine of the angle to the beam axis."""
    ang = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.exp(-(ang / half_angle) ** 2)


class PulsarRadiation:
    def __init__(self, ns, inclination=OBSERVER_INCLINATION, half_angle=BEAM_HALF_ANGLE):
        self.ns = ns
        self.inclination = inclination
        self.half_angle = half_angle
        self.I_rot = self.I_acc = 0.0
        self.radio_on = False

    # ---------------------------------------------------------------- state
    def update(self, tq, state):
        """Recompute channel intensities from the torque/luminosity dict."""
        self.radio_on = state == states.PULSAR_DOMINATED
        self.I_rot = rotation_intensity(tq["L_sd"], state)
        self.I_acc = accretion_intensity(tq["L_acc"], self.ns.mass, self.ns.radius)

    @property
    def intensity(self):
        return min(1.0, self.I_rot + self.I_acc)

    # ------------------------------------------------------------- geometry
    def axis(self, phase=None):
        return magnetic_axis(self.ns.spin_phase if phase is None else phase, self.ns.alpha)

    def directions(self):
        m = self.axis()
        return m, -m

    # --------------------------------------------------------------- pulses
    @property
    def pulse_period(self):
        return self.ns.period

    @property
    def pulse_frequency(self):
        return self.ns.frequency

    def beam_visible(self, sign=+1):
        """Does the +m / -m beam sweep across the observer's line of sight?"""
        colat = self.ns.alpha if sign > 0 else math.pi - self.ns.alpha
        return abs(colat - self.inclination) < self.half_angle

    @property
    def pulses_per_rotation(self):
        return int(self.beam_visible(+1)) + int(self.beam_visible(-1))

    def time_to_next_pulse(self):
        """Seconds (simulated) until the +m beam azimuth next crosses the observer azimuth (0)."""
        if self.ns.omega <= 0.0:
            return math.inf
        return ((-self.ns.spin_phase) % (2.0 * math.pi)) / self.ns.omega

    def profile(self, phases):
        """Observed relative intensity for an array of spin phases."""
        phases = np.asarray(phases, dtype=float)
        sa, ca = math.sin(self.ns.alpha), math.cos(self.ns.alpha)
        m = np.stack([sa * np.cos(phases), sa * np.sin(phases), np.full_like(phases, ca)], axis=1)
        n = observer_direction(self.inclination)
        c = m @ n
        return np.clip(self.intensity * (beam_pattern(c, self.half_angle) + beam_pattern(-c, self.half_angle)), 0.0, 1.0)

    def profile_window(self, n_points, n_rotations):
        """Profile over the last n_rotations ending at the current phase."""
        k = np.arange(n_points)
        phases = self.ns.spin_phase - 2 * np.pi * n_rotations * (1.0 - k / (n_points - 1))
        return phases, self.profile(phases)

    def telemetry(self):
        m = self.axis()
        return {
            "beam_intensity": self.intensity,
            "radio_pulsar_on": self.radio_on,
            "beam_axis": m,
            "beam_half_angle_deg": math.degrees(self.half_angle),
            "observer_inclination_deg": math.degrees(self.inclination),
            "pulse_period_ms": self.pulse_period * 1e3,
            "pulse_frequency_hz": self.pulse_frequency,
            "pulses_per_rotation": self.pulses_per_rotation,
            "time_to_next_pulse_ms": self.time_to_next_pulse() * 1e3,
        }
