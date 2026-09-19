"""
Provenance registry: what is OBSERVED, what is COMPUTED, what is a SIMPLIFIED
MODEL assumption.  Machine-readable; used by `python sim.py --provenance`,
by the --check suite (every telemetry key must be classified, every observed
value must carry a source and match config.py) and to generate the README
table.

Types
    OBSERVED    a published measurement of PSR J1023+0038 (value + citation);
                "OBSERVED-DERIVED" marks quantities inferred from timing under
                standard assumptions (spin-down power, dipole field estimate)
    COMPUTED    evolves from the equations of the model at run time
    SIMPLIFIED  a model assumption or constant chosen for this simulation
                (never presented as a measurement)

Observational sources
    [A09] Archibald et al. 2009, Science 324, 1411  - radio MSP discovery: spin period
    [D12] Deller et al. 2012, ApJ 756, L25          - neutron-star mass
    [A13] Archibald et al. 2013, arXiv:1311.5161    - spin-down power, dipole B
                                                     estimate (Shklovskii-corrected)
    [S14] Stappers et al. 2014, ApJ 790, 39         - 2013 disappearance of the radio
    [P14] Patruno et al. 2014, ApJ 781, L3            pulsar / accretion-disk state
Values below are those adopted from these papers; uncertainties are quoted
where the papers give them.  Nothing else in this project is an observation.
"""
import config as cfg

OBSERVED, OBSERVED_DERIVED, COMPUTED, SIMPLIFIED = "OBSERVED", "OBSERVED-DERIVED", "COMPUTED", "SIMPLIFIED MODEL"

# --- inputs -----------------------------------------------------------------
# (name, type, config attribute or None, value, units, source / model, note)
INPUTS = [
    ("Neutron-star spin period", OBSERVED, "NS_SPIN_PERIOD_OBS", cfg.NS_SPIN_PERIOD_OBS * 1e3, "ms",
     "[A09] radio timing", "1.69 ms"),
    ("Neutron-star mass", OBSERVED, "NS_MASS_OBS", cfg.NS_MASS_OBS / cfg.M_SUN, "Msun",
     "[D12] orbital solution", "1.71 +/- 0.16 Msun"),
    ("Spin-down power", OBSERVED_DERIVED, "SPINDOWN_LUMINOSITY_OBS", cfg.SPINDOWN_LUMINOSITY_OBS * 1e7, "erg/s",
     "[A13] from P, Pdot with I = 1e45 g cm^2", "~4.4e34 erg/s, Shklovskii-corrected"),
    ("Surface dipole field (equatorial)", OBSERVED_DERIVED, "B_SURFACE_OBS", cfg.B_SURFACE_OBS / cfg.GAUSS, "G",
     "[A13] dipole spin-down estimate", "~1e8 G; used as model input B0"),
    ("Observed states", OBSERVED, None, None, "-",
     "[A09][S14][P14]", "radio-pulsar state (pre-2013) and sub-luminous disk state (2013-)"),
    ("Neutron-star radius", SIMPLIFIED, "NS_RADIUS", cfg.NS_RADIUS / 1e3, "km", "assumed canonical value", "not measured for J1023"),
    ("Magnetic misalignment angle", SIMPLIFIED, "MAGNETIC_MISALIGNMENT", None, "deg", "assumed", "35 deg; needed for a beam sweep"),
    ("Moment of inertia", SIMPLIFIED, "MOMENT_OF_INERTIA_PREFACTOR", cfg.MOMENT_OF_INERTIA_PREFACTOR, "-", "I = k M R^2, uniform sphere", "k = 2/5"),
    ("Magnetospheric radius factor", SIMPLIFIED, "XI_MAGNETOSPHERE", cfg.XI_MAGNETOSPHERE, "-", "r_m = xi r_A", "standard 0.5"),
    ("Disk outer radius", SIMPLIFIED, "DISK_OUTER_RADIUS", cfg.DISK_OUTER_RADIUS / 1e3, "km", "fixed disk edge where matter is supplied", "3e5 km, assumed"),
    ("Disk inflow time", SIMPLIFIED, "DISK_VISCOUS_TIME", cfg.DISK_VISCOUS_TIME / cfg.DAY, "d", "alpha-disk scaling t ~ r^1.5, total normalised", "1 day"),
    ("Maximum transfer rate (100 %)", SIMPLIFIED, "MAX_TRANSFER_RATE", cfg.MAX_TRANSFER_RATE * cfg.YEAR / cfg.M_SUN, "Msun/yr", "slider range, cubic mapping", "1e-9 Msun/yr, sub-Eddington"),
    ("Field-burial mass scale", SIMPLIFIED, "FIELD_BURIAL_MASS", cfg.FIELD_BURIAL_MASS / cfg.M_SUN, "Msun", "B = B0/(1 + dM/M_B) (Shibazaki et al. 1989 form)", "1e-3 Msun"),
    ("Maximum neutron-star mass", SIMPLIFIED, "TOV_MASS_LIMIT", cfg.TOV_MASS_LIMIT / cfg.M_SUN, "Msun", "assumed EOS limit", "2.2 Msun"),
    ("State thresholds (fastness)", SIMPLIFIED, "STATE_ACCRETING_MAX_FASTNESS", cfg.STATE_ACCRETING_MAX_FASTNESS, "-", "ACCRETING < 0.85 <= TRANSITION <= 1.15 < PROPELLER", "chosen bands"),
    ("Beam half-angle", SIMPLIFIED, None, 12.0, "deg", "Gaussian beam pattern", "physics/radiation.py"),
    ("Observer inclination", SIMPLIFIED, None, 40.0, "deg", "assumed line of sight", "physics/radiation.py"),
    ("Radio quench factor with disk", SIMPLIFIED, "ACTIVITY_QUENCH_WITH_DISK", cfg.ACTIVITY_QUENCH_WITH_DISK, "-", "observed-inspired (radio pulsar hidden in disk state)", "constant factor"),
    ("Polar outflow: accretion-power fraction", SIMPLIFIED, "ACTIVITY_JET_FRACTION", cfg.ACTIVITY_JET_FRACTION, "-", "P_polar = L_sd + f L_acc", "0.1"),
    ("Polar wind speed", SIMPLIFIED, None, 0.3, "c", "assumed bulk speed", "physics/outflow.py"),
]

# --- computed quantities: telemetry key -> (name, units, equation / model) ------
COMPUTED_KEYS = {
    "mass_msun": ("Neutron-star mass", "Msun", "dM/dt = f_acc Mdot_disk"),
    "accreted_mass_msun": ("Accreted mass", "Msun", "integral of dM/dt"),
    "J": ("Angular momentum", "kg m^2/s", "dJ/dt = Jdot_acc - Jdot_loss"),
    "jdot_net": ("Net torque", "kg m^2/s^2", "Jdot_acc - Jdot_loss"),
    "freq_hz": ("Rotation frequency", "Hz", "f = 1/P, P = 2 pi I / J"),
    "period_ms": ("Rotation period", "ms", "P = 2 pi / Omega"),
    "delta_f_hz": ("Frequency change since start", "Hz", "f - f_obs"),
    "fdot_hz_s": ("Frequency derivative", "Hz/s", "(Jdot/I - Omega Mdot/M) / 2 pi"),
    "E_rot_J": ("Rotational energy", "J", "1/2 I Omega^2"),
    "B_gauss": ("Surface field (evolved)", "G", "B0 / (1 + dM/M_B)"),
    "B_surface_G": ("Surface field, equatorial", "G", "= B_gauss"),
    "B_surface_T": ("Surface field, equatorial", "T", "= B_gauss"),
    "B_polar_G": ("Surface field, polar", "G", "2 B_eq"),
    "B_initial_G": ("Initial field", "G", "input B0"),
    "dipole_moment_A_m2": ("Dipole moment", "A m^2", "4 pi B R^3 / mu0"),
    "r_m_km": ("Magnetospheric radius", "km", "xi (2 pi B^2 R^6 / (mu0 Mdot sqrt(2GM)))^(2/7)"),
    "r_co_km": ("Corotation radius", "km", "(GM/Omega^2)^(1/3)"),
    "r_lc_km": ("Light-cylinder radius", "km", "c / Omega"),
    "fastness": ("Fastness parameter", "-", "(r_m/r_co)^(3/2)"),
    "L_acc_W": ("Accretion luminosity", "W", "G M Mdot_ns / R"),
    "L_sd_W": ("Spin-down luminosity", "W", "(8 pi/3) B^2 R^6 Omega^4 sin^2 alpha / (mu0 c^3)"),
    "L_total_W": ("Total modelled power", "W", "L_acc + L_sd"),
    "activity": ("Activity index", "0-1", "log_index(L_acc + L_sd)"),
    "activity_accretion": ("Accretion activity index", "0-1", "log_index(L_acc)"),
    "activity_rotation": ("Rotation activity index", "0-1", "log_index(L_sd)"),
    "activity_disk": ("Disk activity index", "0-1", "log_index(L_disk)"),
    "mdot_in_msun_yr": ("Mass supply rate at the disk edge", "Msun/yr", "Mdot_max control^3 (control = user input)"),
    "mdot_ns_msun_yr": ("Rate onto the star", "Msun/yr", "f_acc Mdot_disk"),
    "disk_mass_kg": ("Disk mass", "kg", "sum of ring masses"),
    "disk_J": ("Disk angular momentum", "kg m^2/s", "sum m_k sqrt(G M r_k)"),
    "disk_r_in_km": ("Disk inner edge", "km", "self-consistent truncation r_k >= r_m(flux_k)"),
    "disk_r_out_km": ("Disk outer edge", "km", "DISK_OUTER_RADIUS (fixed)"),
    "disk_T_max_K": ("Disk peak temperature", "K", "Shakura-Sunyaev T(r) from local flux"),
    "disk_T_outer_K": ("Disk outer temperature", "K", "Shakura-Sunyaev T(r)"),
    "disk_L_W": ("Disk luminosity", "W", "G M Mdot_mag / (2 r_in)"),
    "disk_omega_in_rad_s": ("Inner-disk angular velocity", "rad/s", "sqrt(G M / r^3)"),
    "disk_jdot_in": ("Disk: angular momentum in", "kg m^2/s^2", "Mdot_in sqrt(G M r_out)"),
    "disk_jdot_mag": ("Disk: angular momentum to magnetosphere", "kg m^2/s^2", "Mdot_mag sqrt(G M r_in)"),
    "disk_jdot_tidal": ("Disk: angular momentum carried outward", "kg m^2/s^2", "budget remainder (leaves the modelled system)"),
    "disk_active_rings": ("Rings holding a disk", "-", "r_k >= r_in"),
    "state": ("System state", "-", "criteria on Mdot_disk, r_m, r_lc, omega_s"),
    "state_criteria": ("Active state criterion", "-", "text of the rule"),
    "state_time_s": ("Time in state", "s", "clock"),
    "state_transitions": ("Number of transitions", "-", "count"),
    "state_last_transition": ("Last transition", "-", "(time, from, to)"),
    "state_direction": ("Transition direction", "-", "+1 toward accretion"),
    "feeding": ("Feeding label", "-", "UI descriptor of the control (not a physical state)"),
    "control": ("Accretion control", "0-1", "user input"),
    "stability": ("Stability index", "-", "max(M/M_TOV, Omega/Omega_breakup)"),
    "extreme": ("Extreme flag", "-", "stability > 0.9"),
    "sim_time_s": ("Simulation time", "s", "clock"),
    "sim_time_formatted": ("Simulation time", "-", "clock"),
    "sim_time_breakdown": ("Simulation time", "d/h/m/s", "clock"),
    "real_elapsed_s": ("Real time elapsed", "s", "wall clock"),
    "sim_seconds_per_real_second": ("Time warp", "s/s", "user setting"),
    "misalignment_deg": ("Misalignment angle", "deg", "input alpha"),
    "spin_axis": ("Spin axis", "-", "z"),
    "magnetic_axis": ("Magnetic axis", "-", "m(spin_phase, alpha)"),
    "magnetic_axis_azimuth_deg": ("Magnetic-axis azimuth", "deg", "= spin phase"),
    "spin_phase_deg": ("Spin phase", "deg", "integral of Omega dt (kinematic clock)"),
    "closed_field_radius_km": ("Closed-field radius", "km", "min(r_m, r_lc)"),
    "B_at_r_m_G": ("Field at r_m", "G", "dipole B(r)"),
    "beam_intensity": ("Beam intensity", "0-1", "I_rot + I_acc"),
    "beam_I_rotation": ("Rotation-powered beam intensity", "0-1", "L_sd/L_sd,obs x quench"),
    "beam_I_accretion": ("Accretion-powered beam intensity", "0-1", "sqrt(L_acc/L_acc,max)"),
    "radio_pulsar_on": ("Radio pulsar visible", "-", "state == PULSAR_DOMINATED"),
    "beam_axis": ("Beam axis", "-", "= magnetic axis"),
    "beam_half_angle_deg": ("Beam half-angle", "deg", "model parameter"),
    "observer_inclination_deg": ("Observer inclination", "deg", "model parameter"),
    "pulse_period_ms": ("Pulse period", "ms", "= P"),
    "pulse_frequency_hz": ("Pulse frequency", "Hz", "= f"),
    "pulses_per_rotation": ("Pulses per rotation", "-", "beam/observer geometry"),
    "time_to_next_pulse_ms": ("Time to next pulse", "ms", "phase / Omega"),
    "outflow_polar_power_W": ("Polar outflow power", "W", "L_sd + f_jet L_acc"),
    "outflow_polar_index": ("Polar outflow index", "0-1", "log_index(P_polar)"),
    "outflow_polar_rotation_fraction": ("Rotation-powered fraction", "-", "L_sd / P_polar"),
    "outflow_cap_angle_deg": ("Polar-cap angle", "deg", "asin sqrt(R/r_lc)"),
    "outflow_polar_speed_c": ("Polar wind speed", "c", "model parameter"),
    "outflow_eq_mdot_msun_yr": ("Propeller ejection rate", "Msun/yr", "Mdot_disk - Mdot_ns"),
    "outflow_eq_index": ("Ejection index", "0-1", "(Mdot_ej/Mdot_max)^(1/3)"),
    "outflow_eq_speed_km_s": ("Ejection speed", "km/s", "sqrt(2 G M / r_m)"),
}

# telemetry keys that are user settings / bookkeeping rather than physics
SETTING_KEYS = {"control", "feeding", "sim_seconds_per_real_second", "real_elapsed_s", "sim_time_formatted", "sim_time_breakdown"}


def classify(key):
    """Type of a telemetry key."""
    if key in SETTING_KEYS:
        return "USER SETTING"
    if key in COMPUTED_KEYS:
        return COMPUTED
    return None


def observed_inputs():
    return [e for e in INPUTS if e[1] in (OBSERVED, OBSERVED_DERIVED)]


def simplified_inputs():
    return [e for e in INPUTS if e[1] == SIMPLIFIED]


PRIMARY_COMPUTED = ["mass_msun", "J", "freq_hz", "period_ms", "fdot_hz_s", "E_rot_J", "B_gauss", "r_m_km", "r_co_km", "r_lc_km",
                    "fastness", "L_acc_W", "L_sd_W", "activity", "mdot_ns_msun_yr", "disk_mass_kg", "disk_T_max_K",
                    "state", "pulse_frequency_hz", "outflow_polar_index"]


def table_rows(concise=False):
    """Rows for the README: (quantity, type, source/model, units).
    concise=True lists the inputs plus the principal computed quantities."""
    rows = []
    for name, typ, attr, value, units, source, note in INPUTS:
        val = "" if value is None else f" = {value:g}"
        rows.append((name + val, typ, source + (f" ({note})" if note else ""), units))
    for key, (name, units, model) in COMPUTED_KEYS.items():
        if key in SETTING_KEYS or (concise and key not in PRIMARY_COMPUTED):
            continue
        rows.append((f"{name} [{key}]", COMPUTED, model, units))
    return rows


def format_table(markdown=True, concise=False):
    rows = table_rows(concise)
    if markdown:
        out = ["| Quantity | Type | Source / Model | Units |", "|---|---|---|---|"]
        out += [f"| {q} | {t} | {s} | {u} |" for q, t, s, u in rows]
        return "\n".join(out)
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    out = [f"{'Quantity':<{w[0]}}  {'Type':<{w[1]}}  {'Source / Model':<{w[2]}}  Units"]
    out += [f"{q:<{w[0]}}  {t:<{w[1]}}  {s:<{w[2]}}  {u}" for q, t, s, u in rows]
    return "\n".join(out)
