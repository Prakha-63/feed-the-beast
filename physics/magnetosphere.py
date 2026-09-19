"""
Simplified magnetic field: an idealized, rigidly rotating dipole.

    rotation axis   z_hat (fixed; the spin axis of the neutron star)
    magnetic axis   m(phi) = ( sin(alpha) cos(phi), sin(alpha) sin(phi), cos(alpha) )
                    alpha = misalignment angle (config.MAGNETIC_MISALIGNMENT,
                    per-star NeutronStar.alpha), phi = spin phase, which the
                    neutron-star model advances with Omega = J/I.  The axis is
                    therefore a *function of the rotation state*, never an
                    independent animation.
    convention      B_s = surface field at the magnetic EQUATOR (the value that
                    enters the spin-down and Alfven-radius formulas via
                    mu = B_s R^3 in cgs); the polar field is 2 B_s.
    dipole moment   mu = 4 pi B_s R^3 / mu0            (SI:  B_eq(r) = mu0 mu / (4 pi r^3))
    field strength  |B|(r, theta) = B_s (R/r)^3 sqrt(1 + 3 cos^2 theta)
                    (theta = magnetic colatitude)
    field line      r(theta) = L sin^2(theta)   (shell parameter L = equatorial radius)
                    lines start/end on the surface at sin^2(theta0) = R / L
    closed region   1.05 R <= L <= min(r_m, r_lc): the disk truncates the magnetosphere
                    at r_m, the light cylinder at r_lc = c / Omega

MODEL PARAMETERS (documented, not measured for J1023 individually):
    B_SURFACE_OBS       1e8 G  - dipole estimate from the observed spin-down
    MAGNETIC_MISALIGNMENT  35 deg - assumed (needed for a pulsar beam sweep)
    field burial        B = B0 / (1 + dM_acc / M_B)  (neutron_star.py)

NOT modelled: multipoles, plasma loading / force-free corrections, field
line opening beyond r_lc, MHD interaction with the disk.
"""
import math

import numpy as np

import config as cfg

SPIN_AXIS = np.array([0.0, 0.0, 1.0])


def magnetic_axis(spin_phase, alpha):
    return np.array([math.sin(alpha) * math.cos(spin_phase), math.sin(alpha) * math.sin(spin_phase), math.cos(alpha)])


def magnetic_basis(spin_phase, alpha):
    """Orthonormal basis (e1, e2, m) with m the magnetic axis; e1, e2 span the
    magnetic equatorial plane (e2 along z x m, so e1 lies in the m-z plane)."""
    m = magnetic_axis(spin_phase, alpha)
    e2 = np.cross(SPIN_AXIS, m)
    if np.linalg.norm(e2) < 1e-9:              # aligned rotator: any perpendicular pair
        e2 = np.array([0.0, 1.0, 0.0])
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, m)
    return e1, e2, m


def dipole_moment(B_surface, R):
    """SI dipole moment from the equatorial surface field."""
    return 4.0 * math.pi * B_surface * R ** 3 / cfg.MU0


def field_strength(B_surface, R, r, theta):
    """|B| of a dipole at radius r and magnetic colatitude theta."""
    r = max(r, R)
    return B_surface * (R / r) ** 3 * math.sqrt(1.0 + 3.0 * math.cos(theta) ** 2)


def field_vector(B_surface, R, pos, basis):
    """Dipole B-vector at position pos (m, NS-centred) for the given basis."""
    e1, e2, m = basis
    r = np.linalg.norm(pos)
    if r < R:
        r = R
    rhat = pos / np.linalg.norm(pos)
    # B = B_s (R/r)^3 [3 (m.rhat) rhat - m]   with B_s the equatorial surface value
    return B_surface * (R / r) ** 3 * (3.0 * np.dot(m, rhat) * rhat - m)


def field_line(L, R, n_points, phi=0.0):
    """Points of one dipole field line in magnetic coordinates (theta, r):
    r = L sin^2 theta from the northern to the southern footpoint."""
    if L <= R:
        return np.zeros((0, 2))
    th0 = math.asin(math.sqrt(R / L))
    th = np.linspace(th0, math.pi - th0, n_points)
    r = L * np.sin(th) ** 2
    return np.stack([th, r], axis=1)


def field_line_points(L, R, n_points, phi, basis):
    """3-D points (m) of a dipole field line at magnetic azimuth phi."""
    e1, e2, m = basis
    tr = field_line(L, R, n_points)
    th, r = tr[:, 0], tr[:, 1]
    d = (np.sin(th) * np.cos(phi))[:, None] * e1 + (np.sin(th) * np.sin(phi))[:, None] * e2 + np.cos(th)[:, None] * m
    return d * r[:, None]


def shell_parameters(R, r_max, n_shells, r_min_factor=1.05):
    """Log-spaced shell radii for the closed magnetosphere between 1.05 R and r_max."""
    r_max = max(r_max, r_min_factor * R * 1.01)
    return r_min_factor * R * (r_max / (r_min_factor * R)) ** ((np.arange(n_shells) + 1.0) / n_shells)


class Magnetosphere:
    """Read-only view of the field geometry for the current neutron-star state."""

    def __init__(self, ns):
        self.ns = ns

    @property
    def alpha(self):
        return self.ns.alpha

    @property
    def spin_axis(self):
        return SPIN_AXIS.copy()

    @property
    def magnetic_axis(self):
        return magnetic_axis(self.ns.spin_phase, self.ns.alpha)

    @property
    def basis(self):
        return magnetic_basis(self.ns.spin_phase, self.ns.alpha)

    @property
    def B_surface(self):
        return self.ns.B

    @property
    def dipole_moment(self):
        return dipole_moment(self.ns.B, self.ns.radius)

    def closed_region_radius(self, r_m, r_lc):
        return min(r_m, r_lc)

    def telemetry(self, r_m, r_lc):
        m = self.magnetic_axis
        return {
            "B_surface_T": self.ns.B,
            "B_surface_G": self.ns.B / cfg.GAUSS,          # equatorial surface field
            "B_polar_G": 2.0 * self.ns.B / cfg.GAUSS,
            "B_initial_G": self.ns.B0 / cfg.GAUSS,
            "dipole_moment_A_m2": self.dipole_moment,
            "misalignment_deg": math.degrees(self.ns.alpha),
            "spin_axis": SPIN_AXIS.copy(),
            "magnetic_axis": m,
            "magnetic_axis_azimuth_deg": math.degrees(math.atan2(m[1], m[0])) % 360.0,
            "spin_phase_deg": math.degrees(self.ns.spin_phase),
            "closed_field_radius_km": self.closed_region_radius(r_m, r_lc) / 1e3,
            "B_at_r_m_G": field_strength(self.ns.B, self.ns.radius, r_m, math.pi / 2) / cfg.GAUSS if math.isfinite(r_m) else 0.0,
        }
