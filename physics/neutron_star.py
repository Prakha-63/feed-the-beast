"""
Neutron-star state and the equations that connect it.

Everything here is pure Python/numpy scalars in SI.  The system is small
(a handful of ODEs) so a Taichi kernel would add nothing.

Equations (all documented in README):
    I       = k M R^2                                 (uniform-sphere approx)
    Omega   = J / I
    P       = 2 pi / Omega
    f       = 1 / P
    E_rot   = 1/2 I Omega^2
    r_A     = ( 2 pi B^2 R^6 / (mu0 Mdot sqrt(2 G M)) )^(2/7)   Alfven radius, SI
    r_m     = max(R, xi r_A)                          magnetospheric radius
    r_co    = (G M / Omega^2)^(1/3)                   corotation radius
    r_lc    = c / Omega                               light-cylinder radius
    omega_s = (r_m / r_co)^(3/2)                      fastness parameter
    Jdot_acc = Mdot sqrt(G M r_m) (1 - omega_s)       accretion torque (Ghosh-Lamb-like)
    L_sd    = (8 pi / 3) B^2 R^6 Omega^4 sin^2(alpha) / (mu0 c^3)   dipole spin-down
    Jdot_md = L_sd / Omega                            magnetic-dipole braking torque
    L_acc   = G M Mdot_ns / R                         accretion luminosity
    B       = B0 / (1 + dM_acc / M_B)                 field burial by accreted mass
"""
import math

import config as cfg


def moment_of_inertia(mass, radius):
    """I = k M R^2 (uniform sphere, k = 2/5).  Raises on non-physical input."""
    if mass <= 0.0 or radius <= 0.0:
        raise ValueError("moment_of_inertia: mass and radius must be positive")
    return cfg.MOMENT_OF_INERTIA_PREFACTOR * mass * radius ** 2


def alfven_radius(B, R, M, mdot):
    """Alfven radius in SI (ram pressure = magnetic pressure of a dipole)."""
    if mdot <= 0.0:
        return math.inf
    return (2.0 * math.pi * B * B * R ** 6 / (cfg.MU0 * mdot * math.sqrt(2.0 * cfg.G * M))) ** (2.0 / 7.0)


def magnetospheric_radius(B, R, M, mdot):
    return max(R, cfg.XI_MAGNETOSPHERE * alfven_radius(B, R, M, mdot))


def corotation_radius(M, omega):
    """Infinite for a non-rotating star (no corotation point)."""
    if omega <= 0.0:
        return math.inf
    return (cfg.G * M / (omega * omega)) ** (1.0 / 3.0)


def light_cylinder_radius(omega):
    if omega <= 0.0:
        return math.inf
    return cfg.C / omega


def spindown_luminosity(B, R, omega, alpha):
    if omega <= 0.0:
        return 0.0
    return (8.0 * math.pi / 3.0) * B * B * R ** 6 * omega ** 4 * math.sin(alpha) ** 2 / (cfg.MU0 * cfg.C ** 3)


def accretion_fraction(fastness):
    """Fraction of disk matter reaching the surface: 1 for slow rotators,
    -> 0 in the propeller regime.  Smooth logistic switch around omega_s = 1."""
    x = (fastness - 1.0) / cfg.FASTNESS_TRANSITION_WIDTH
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(x))


def accretion_torque(mdot, M, r_m, fastness):
    """SIMPLIFIED accretion torque (Ghosh & Lamb-like):
        Jdot_acc = Mdot * sqrt(G M r_m) * (1 - omega_s)
    Matter arriving at r_m carries the Keplerian specific angular momentum
    sqrt(G M r_m); the factor (1 - omega_s) changes sign at the corotation
    radius so fast rotators are braked (propeller).  No disk-field coupling
    beyond this factor, no warping, no radiative feedback."""
    if mdot <= 0.0 or not math.isfinite(r_m) or not math.isfinite(fastness):
        return 0.0
    return mdot * math.sqrt(cfg.G * M * r_m) * (1.0 - fastness)


def loss_torque(B, R, omega, alpha):
    """SIMPLIFIED angular-momentum loss: vacuum magnetic-dipole braking only,
        Jdot_loss = L_sd / Omega.
    Gravitational-wave and wind-braking channels are neglected."""
    if omega <= 0.0:
        return 0.0
    return spindown_luminosity(B, R, omega, alpha) / omega


class NeutronStar:
    """Neutron-star state.

    Numerical note: M ~ 3e30 kg and J ~ 7e41 kg m^2/s, while one integration
    substep changes them by ~1e16 kg and ~1e28 kg m^2/s.  Adding such
    increments directly to the totals loses them to round-off (the ulp of M is
    ~7e14 kg).  The state is therefore stored as  reference value + accumulated
    change  and the totals are recomputed on read, which keeps every increment
    to full double precision at any time-warp.
    """

    def __init__(self, mass=cfg.NS_MASS_OBS, radius=cfg.NS_RADIUS, B0=cfg.B_SURFACE_OBS,
                 alpha=cfg.MAGNETIC_MISALIGNMENT, J=0.0, accreted_mass=0.0, spin_phase=0.0):
        self.radius = radius
        self.B0 = B0
        self.alpha = alpha
        self.accreted_mass = accreted_mass   # kg accreted since start (mass gain + field burial)
        self._mass0 = mass - accreted_mass   # reference mass (kg)
        self.spin_phase = spin_phase         # rad, visual phase about the spin axis
        self._J0 = 0.0
        self._dJ = 0.0
        if J == 0.0:                         # default: observed spin period
            J = self.inertia * 2.0 * math.pi / cfg.NS_SPIN_PERIOD_OBS
        self._J0 = J

    # --- primary state ----------------------------------------------------------
    @property
    def mass(self):
        """M = M_0 + accreted mass  (kg)."""
        return self._mass0 + self.accreted_mass

    @mass.setter
    def mass(self, value):
        self._mass0 = value - self.accreted_mass

    @property
    def J(self):
        """Angular momentum  J = J_0 + accumulated torque impulse  (kg m^2/s)."""
        return self._J0 + self._dJ

    @J.setter
    def J(self, value):
        self._J0, self._dJ = value, 0.0

    def __repr__(self):
        return f"NeutronStar(mass={self.mass:.6e}, J={self.J:.6e}, accreted_mass={self.accreted_mass:.3e})"

    # --- derived quantities -------------------------------------------------
    @property
    def inertia(self):
        return moment_of_inertia(self.mass, self.radius)

    @property
    def omega(self):
        """Omega = J / I  (rad/s).  J is kept >= 0, so Omega >= 0."""
        return max(self.J, 0.0) / self.inertia

    @property
    def period(self):
        """P = 2 pi / Omega; infinite for a non-rotating star."""
        om = self.omega
        return math.inf if om <= 0.0 else 2.0 * math.pi / om

    @property
    def frequency(self):
        """f = 1 / P  (Hz); zero for a non-rotating star."""
        P = self.period
        return 0.0 if not math.isfinite(P) else 1.0 / P

    @property
    def rotational_energy(self):
        return 0.5 * self.inertia * self.omega ** 2

    @property
    def B(self):
        return self.B0 / (1.0 + self.accreted_mass / cfg.FIELD_BURIAL_MASS)

    @property
    def breakup_omega(self):
        return math.sqrt(cfg.G * self.mass / self.radius ** 3)

    # --- torques --------------------------------------------------------------
    def torques(self, mdot_disk):
        """Return dict of instantaneous quantities for a given disk feeding rate
        (kg/s delivered to the magnetosphere)."""
        mdot_disk = max(0.0, mdot_disk) if math.isfinite(mdot_disk) else 0.0   # safeguard
        omega = self.omega
        B = self.B
        r_m = magnetospheric_radius(B, self.radius, self.mass, mdot_disk)
        r_co = corotation_radius(self.mass, omega)
        r_lc = light_cylinder_radius(omega)
        if not math.isfinite(r_m):
            fastness = math.inf          # no disk: nothing to accrete
        elif not math.isfinite(r_co):
            fastness = 0.0               # non-rotating star: always accretes
        else:
            fastness = (r_m / r_co) ** 1.5
        f_acc = accretion_fraction(fastness) if math.isfinite(fastness) else 0.0
        jdot_acc = accretion_torque(mdot_disk, self.mass, r_m, fastness)
        if r_m >= r_lc:
            # ejector regime: the disk is truncated outside the light cylinder,
            # the pulsar wind holds it off and there is no disk-star coupling
            f_acc, jdot_acc = 0.0, 0.0
        L_sd = spindown_luminosity(B, self.radius, omega, self.alpha)
        jdot_md = loss_torque(B, self.radius, omega, self.alpha)
        mdot_ns = f_acc * mdot_disk
        mdot_ejected = mdot_disk - mdot_ns
        L_acc = cfg.G * self.mass * mdot_ns / self.radius
        return dict(r_m=r_m, r_co=r_co, r_lc=r_lc, fastness=fastness, f_acc=f_acc,
                    jdot_acc=jdot_acc, jdot_md=jdot_md, jdot_loss=jdot_md, L_sd=L_sd, L_acc=L_acc,
                    mdot_ns=mdot_ns, mdot_ejected=mdot_ejected)

    def apply(self, dM, dJ):
        """Advance state by finite increments.  Safeguards: accreted mass can
        only grow, J cannot go negative (braking stops at zero spin) and
        non-finite increments are rejected."""
        if not (math.isfinite(dM) and math.isfinite(dJ)):
            raise ValueError("NeutronStar.apply: non-finite increment")
        dM = max(dM, 0.0)
        self.accreted_mass += dM             # mass = mass0 + accreted_mass
        self._dJ += dJ
        if self._J0 + self._dJ < 0.0:
            self._dJ = -self._J0
