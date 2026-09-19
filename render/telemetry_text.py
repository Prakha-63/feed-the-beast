"""
Telemetry text: pure formatting of SystemModel.telemetry() values (no Taichi,
no window) so the HUD content can be unit-tested and reused.

Formatting rules
    fmt_sci(x, unit)   mantissa with 3 significant digits + exponent + unit
    fixed values keep enough decimals to show the changes the physics makes
    (mass to 1e-6 Msun plus the accreted mass in scientific notation, spin
    frequency to 1e-4 Hz plus the change since start, period to 1 ns)
Every line carries its unit.  Nothing here is a literal scientific value:
all numbers come from the telemetry dict.
"""
import math

from physics import states
from physics.simtime import format_duration

WARN = (1.0, 0.25, 0.25)
DIM = (0.6, 0.62, 0.7)


def fmt_sci(x, unit, digits=3):
    if x is None or not math.isfinite(x):
        return f"-- {unit}"
    if x == 0:
        return f"0 {unit}"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if abs(m) >= 9.995:            # rounding overflow (e.g. 9.999 -> 10.0)
        m /= 10.0
        e += 1
    return f"{m:.{digits - 1}f}e{e:+d} {unit}"


def fmt_km(x):
    return f"{x:.1f} km" if math.isfinite(x) else "-- km"


def telemetry_lines(t, warp_label="", kin_capped=False):
    """List of (text, color_or_None) for the telemetry panel."""
    L = []
    col = states.STATE_COLORS[t["state"]]
    L.append((f"STATE        {t['state']}", col))
    L.append((f"             {t['state_criteria']}  for {format_duration(t['state_time_s'])[2:]}  ({t['state_transitions']} trans.)", DIM))
    if t["extreme"]:
        L.append((f"WARNING      EXTREME  stability index {t['stability']:.3f}  (M/M_TOV or f/f_breakup > 0.9)", WARN))
    else:
        L.append((f"STABILITY    {t['stability']:.3f}  (max of M/M_TOV, f/f_breakup)", DIM))
    L.append((f"SIM TIME     {format_duration(t['sim_time_s'])}  [{warp_label}]" + ("  (visual phases capped)" if kin_capped else ""), None))
    L.append(("", None))
    L.append((f"MASS         {t['mass_msun']:.6f} Msun   accreted {fmt_sci(t['accreted_mass_msun'], 'Msun')}", None))
    L.append((f"ACCRETION    supply {fmt_sci(t['mdot_in_msun_yr'], 'Msun/yr')}  -> star {fmt_sci(t['mdot_ns_msun_yr'], 'Msun/yr')}  [{t['feeding']}]", None))
    L.append((f"DISK MASS    {fmt_sci(t['disk_mass_kg'], 'kg')}   inner edge {fmt_km(t['disk_r_in_km'])}   J_disk {fmt_sci(t['disk_J'], 'kg m^2/s')}", None))
    L.append((f"ANG. MOM.    J = {fmt_sci(t['J'], 'kg m^2/s')}   dJ/dt = {fmt_sci(t['jdot_net'], 'kg m^2/s^2')}", None))
    L.append((f"ROTATION     {t['freq_hz']:.4f} Hz   ({t['delta_f_hz']:+.3e} Hz since start)", None))
    L.append((f"PERIOD       {t['period_ms']:.6f} ms   f-dot {t['fdot_hz_s']:+.2e} Hz/s  ({'spin-up' if t['fdot_hz_s'] > 0 else 'spin-down' if t['fdot_hz_s'] < 0 else 'steady'})", None))
    L.append((f"MAG. FIELD   {fmt_sci(t['B_surface_G'], 'G')} (equatorial)   {fmt_sci(t['B_surface_T'], 'T')}   alpha = {t['misalignment_deg']:.0f} deg", DIM))
    L.append((f"ROT. ENERGY  {fmt_sci(t['E_rot_J'], 'J')} = {fmt_sci(t['E_rot_J'] * 1e7, 'erg')}", DIM))
    L.append(("", None))
    L.append((f"ACTIVITY     {t['activity']:.3f} (index 0-1)   L_tot {fmt_sci(t['L_total_W'] * 1e7, 'erg/s')}", None))
    L.append((f"  L_acc {fmt_sci(t['L_acc_W'] * 1e7, 'erg/s')}   L_sd {fmt_sci(t['L_sd_W'] * 1e7, 'erg/s')}   L_disk {fmt_sci(t['disk_L_W'] * 1e7, 'erg/s')}", DIM))
    L.append((f"RADII        r_m {fmt_km(t['r_m_km'])}  r_co {fmt_km(t['r_co_km'])}  r_lc {fmt_km(t['r_lc_km'])}   omega_s {t['fastness']:.2f}" if math.isfinite(t['fastness'])
              else f"RADII        r_m {fmt_km(t['r_m_km'])}  r_co {fmt_km(t['r_co_km'])}  r_lc {fmt_km(t['r_lc_km'])}   omega_s --", DIM))
    L.append((f"OUTFLOW      polar {t['outflow_polar_index']:.2f}  ejection {t['outflow_eq_index']:.2f} (indices)   T_disk,max {fmt_sci(t['disk_T_max_K'], 'K')}", DIM))
    L.append(("rings: orange r_m   cyan r_co   blue r_lc   |   beams: blue rotation-powered, orange accretion-powered", DIM))
    return L


def control_lines(t, warp_label, camera_name, backend, fps):
    return [
        f"feed {100 * t['control']:5.1f} %  ->  mass supply {fmt_sci(t['mdot_in_msun_yr'], 'Msun/yr')}   [{t['feeding']}]",
        f"TIME WARP   {warp_label}     ( [ / ] )",
        f"camera: {camera_name}   1-6 presets  0 recentre",
        "L-drag orbit  R-drag pan  M-drag / UP,DOWN zoom",
        "LEFT/RIGHT feed -/+ 5 %   X starve   F feed 100 %",
        f"SPACE pause  R reset  T restart  ESC quit   {backend}  {fps:4.0f} fps",
    ]
