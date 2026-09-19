"""
Global configuration for "Feed the Beast" — units, reference-system inputs,
model constants and rendering knobs.

UNIT CONVENTIONS (physics side, everything SI unless stated):
    mass            kg          (displayed in solar masses, M_SUN)
    length          m           (displayed in km)
    time            s           (displayed in ms / h / d / yr)
    angular momentum kg m^2/s
    magnetic field  tesla       (displayed in gauss; 1 T = 1e4 G)
    luminosity      watt        (displayed in W and erg/s)
    accretion rate  kg/s        (displayed in M_SUN/yr)

Rendering uses a separate "scene unit" (see render/scene_map.py): the neutron
star radius is 1 scene unit and larger radii are log-compressed.  Scene units
never feed back into physics.
"""
import math

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
G = 6.67430e-11          # m^3 kg^-1 s^-2
C = 2.99792458e8         # m/s
MU0 = 4.0e-7 * math.pi   # vacuum permeability, T m / A
M_SUN = 1.98847e30       # kg
R_SUN = 6.957e8          # m
YEAR = 365.25 * 86400.0  # s
DAY = 86400.0
HOUR = 3600.0
GAUSS = 1.0e-4           # tesla per gauss

# ---------------------------------------------------------------------------
# REAL OBSERVATIONAL INPUTS — PSR J1023+0038 (see README for citations)
# ---------------------------------------------------------------------------
REF_SYSTEM_NAME = "PSR J1023+0038"
NS_SPIN_PERIOD_OBS = 1.6879e-3     # s   (Archibald et al. 2009)
NS_MASS_OBS = 1.71 * M_SUN          # kg  (Deller et al. 2012)
B_SURFACE_OBS = 1.0e8 * GAUSS       # T   (dipole estimate from spin-down, ~1e8 G)
SPINDOWN_LUMINOSITY_OBS = 4.4e34 * 1e-7  # W (4.4e34 erg/s; Archibald et al. 2013)

# ---------------------------------------------------------------------------
# SIMPLIFIED MODEL CONSTANTS (assumptions, all explicit)
# ---------------------------------------------------------------------------
NS_RADIUS = 12.0e3                  # m; canonical value, not measured for J1023
MAGNETIC_MISALIGNMENT = math.radians(35.0)  # angle between spin and magnetic axes (assumed)
MOMENT_OF_INERTIA_PREFACTOR = 0.4   # I = k M R^2, uniform-sphere approximation (k=2/5)
XI_MAGNETOSPHERE = 0.5              # r_m = xi * r_Alfven (standard 0.5)
DISK_VISCOUS_TIME = 1.0 * DAY       # s; total viscous inflow time from the disk edge to the star
DISK_OUTER_RADIUS = 3.0e8           # m; fixed disk outer edge (~0.25 of a J1023-like binary separation; assumed)
MAX_TRANSFER_RATE = 1.0e-9 * M_SUN / YEAR  # kg/s; slider = 100 %  (sub-Eddington)
ACCRETION_FLOOR = 1.0e-15 * M_SUN / YEAR   # kg/s; below this the disk is "gone"
FIELD_BURIAL_MASS = 1.0e-3 * M_SUN  # kg; B = B0/(1 + dM/M_B)  (Shibazaki et al. 1989 form)
TOV_MASS_LIMIT = 2.2 * M_SUN        # kg; assumed maximum NS mass (EOS-dependent, 2.0–2.3)
FASTNESS_TRANSITION_WIDTH = 0.08    # smoothness of accretion/propeller switch in omega_s

# ---------------------------------------------------------------------------
# STATE-TRANSITION MODEL (simulated criteria; see physics/states.py)
# All conditions are evaluated on computed variables only:
#   omega_s = (r_m / r_co)^(3/2)   fastness parameter
#   Mdot_disk                      rate the disk hands to the magnetosphere
#   r_m, r_lc                      magnetospheric / light-cylinder radii
# ---------------------------------------------------------------------------
STATE_ACCRETING_MAX_FASTNESS = 0.85    # ACCRETING      : omega_s <  0.85
STATE_PROPELLER_MIN_FASTNESS = 1.15    # TRANSITION     : 0.85 <= omega_s <= 1.15
#                                        LOW_ACCRETION  : omega_s >  1.15  (propeller)
#                                        PULSAR_DOMINATED: Mdot_disk < ACCRETION_FLOOR  or  r_m >= r_lc (ejector)
STATE_HISTORY_LENGTH = 32              # transitions kept for telemetry

# ---------------------------------------------------------------------------
# ACTIVITY MODEL (physics/activity.py) - dimensionless indices on a log scale
# ---------------------------------------------------------------------------
ACTIVITY_L_LOW = 1.0e26     # W   index 0 at/below this total luminosity
ACTIVITY_L_HIGH = 1.3e31    # W   index 1 at/above (~Eddington for 1.7 Msun)
ACTIVITY_QUENCH_WITH_DISK = 0.25   # rotation-powered emission fraction surviving when a disk is present
ACTIVITY_JET_FRACTION = 0.10       # fraction of accretion power assumed in polar outflow

# ---------------------------------------------------------------------------
# TIME
# ---------------------------------------------------------------------------
# Selectable time-warp factors: simulated seconds per real second.
TIME_WARPS = [
    (1.0e-3, "1 s = 1 ms  (slow-motion)"),
    (1.0e-2, "1 s = 10 ms (slow-motion)"),
    (1.0, "real time"),
    (HOUR, "1 s = 1 hour"),
    (6 * HOUR, "1 s = 6 hours"),
    (DAY, "1 s = 1 day"),
    (YEAR, "1 s = 1 year"),
    (1.0e3 * YEAR, "1 s = 1 kyr"),
    (1.0e5 * YEAR, "1 s = 100 kyr"),
    (1.0e6 * YEAR, "1 s = 1 Myr"),
]
DEFAULT_WARP_INDEX = 4   # 1 s = 6 hours: a feed/starve cycle plays out in ~15 s
# Cap on the kinematic (visual-phase) time advanced per frame.  Secular
# physics always uses the full warped dt; only visual phases (orbit position,
# disk-particle azimuth, spin phase) are limited so the picture stays legible
# at extreme warp.  See README "Simulation time".
KINEMATIC_DT_CAP = 600.0   # s per frame
SECULAR_SUBSTEPS = 8

# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------
WINDOW_RES = (1280, 720)
SCENE_LOG_K = 2.2          # log-compression strength (scene_map.py)
N_DISK_PARTICLES = 14000
N_WIND_PARTICLES = 1600
N_BEAM_PARTICLES = 600
N_STARS = 2500
N_FIELD_LINES = 5          # closed dipole shells drawn
FIELD_LINE_SEGMENTS = 48
STARFIELD_RADIUS = 400.0
DEFAULT_SEED = 42
