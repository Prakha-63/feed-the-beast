"""
Scientific validation of the core equations:  python sim.py --check

Sections: core rotation identities, mass, angular momentum / torque, disk,
zero feeding, simulation time, magnetic axis + beams, outflow, states, the
feed/starve chain, stability, provenance, and a CPU-kernel causality audit.
Each check prints PASS/FAIL with the measured error and its tolerance:
    identities (Omega=J/I, P=2pi/Omega, f=1/P, E_rot)   1e-12 relative
    exact-solution steps (mass, dJ over one step)       1e-9 relative / 1e-15 |J|
    multi-step integration                              1e-3 relative
    Taichi f32 kernels                                  2e-3 relative
"""
import math
import os

import numpy as np

import config as cfg
from physics.neutron_star import NeutronStar, alfven_radius, magnetospheric_radius
from physics.disk import AccretionDisk
from physics.model import SystemModel
from physics import states

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} {detail}")
    return ok


def section(title):
    print(f"\n{title}")


def steady_model(control, hours=400):
    """Model with the disk relaxed to steady state (total inflow time = 1 day)."""
    m = SystemModel()
    m.set_control(control)
    for _ in range(hours):
        m.advance(cfg.HOUR)
    return m


# ---------------------------------------------------------------------------
def check_rotation():
    section("CORE ROTATION IDENTITIES  Omega = J/I, P = 2pi/Omega, f = 1/P, E = 1/2 I Omega^2")
    m = steady_model(1.0)
    worst = dict(omega=0.0, P=0.0, f=0.0, E=0.0)
    for _ in range(40):
        m.advance(1e4 * cfg.YEAR)
        ns = m.ns
        om = ns.J / ns.inertia
        worst["omega"] = max(worst["omega"], abs(ns.omega - om) / om)
        worst["P"] = max(worst["P"], abs(ns.period - 2 * math.pi / om) / ns.period)
        worst["f"] = max(worst["f"], abs(ns.frequency * ns.period - 1.0))
        worst["E"] = max(worst["E"], abs(ns.rotational_energy - 0.5 * ns.inertia * om ** 2) / ns.rotational_energy)
    for k, label in (("omega", "Omega = J / I"), ("P", "P = 2 pi / Omega"), ("f", "f = 1 / P"), ("E", "E_rot = 1/2 I Omega^2")):
        report(label + " along a fed trajectory", worst[k] < 1e-12, f"max rel err={worst[k]:.1e}")
    ref = NeutronStar()
    report("initial P equals the observed 1.6879 ms", abs(ref.period - cfg.NS_SPIN_PERIOD_OBS) < 1e-15 * cfg.NS_SPIN_PERIOD_OBS, f"P={ref.period * 1e3:.5f} ms")
    report("I = k M R^2 with k = 2/5", abs(ref.inertia / (0.4 * ref.mass * ref.radius ** 2) - 1) < 1e-14, f"I={ref.inertia:.3e} kg m^2")


def check_mass():
    section("MASS   dM/dt = f_acc * Mdot_disk")
    m = steady_model(1.0)
    rel = abs(m.mdot_disk / m.mdot_in - 1)
    report("disk reaches steady state: Mdot_disk = supply rate", rel < 1e-6, f"rel err={rel:.1e}")
    M0, T = m.ns.mass, 1000.0 * cfg.YEAR
    m.advance(T)
    err = abs((m.ns.mass - M0 - m.tq["f_acc"] * m.mdot_in * T) / (m.tq["f_acc"] * m.mdot_in * T))
    report("dM = f_acc * Mdot * dt over 1000 yr", err < 1e-6, f"rel err={err:.2e}")
    p = steady_model(0.15)
    report("mass budget: Mdot_ns + Mdot_ejected = Mdot_disk (propeller)", p.tq["mdot_ns"] + p.tq["mdot_ejected"] == p.mdot_disk and p.tq["f_acc"] < 0.5,
           f"f_acc={p.tq['f_acc']:.3f} at omega_s={p.tq['fastness']:.2f}")


def check_angular_momentum():
    section("ANGULAR MOMENTUM   dJ/dt = Jdot_acc - Jdot_loss")
    ns = NeutronStar()
    mdot = 0.5 * cfg.MAX_TRANSFER_RATE
    tq = ns.torques(mdot)
    J0, dt = ns.J, cfg.YEAR
    ns.apply(tq["mdot_ns"] * dt, (tq["jdot_acc"] - tq["jdot_loss"]) * dt)
    err = abs(ns.J - J0 - (tq["jdot_acc"] - tq["jdot_loss"]) * dt) / J0
    report("single step: dJ = (Jdot_acc - Jdot_loss) dt", err < 1e-15, f"|err|/J={err:.1e}")
    m = steady_model(1.0)
    J0, T = m.ns.J, 100.0 * cfg.YEAR
    jd0 = m.tq["jdot_acc"] - m.tq["jdot_loss"]
    for _ in range(100):
        m.advance(T / 100)
    jd1 = m.tq["jdot_acc"] - m.tq["jdot_loss"]
    err = abs((m.ns.J - J0) - 0.5 * (jd0 + jd1) * T) / abs(m.ns.J - J0)
    report("100 steps: dJ matches the trapezoidal torque integral", err < 1e-3, f"rel err={err:.2e}")
    ref = NeutronStar()
    r_m = ref.torques(mdot)["r_m"]
    om_eq = math.sqrt(cfg.G * ref.mass / r_m ** 3)
    slow, fast, eq = (NeutronStar(J=ref.inertia * om_eq * k).torques(mdot) for k in (0.5, 1.5, 1.0))
    report("torque: spin-up below, spin-down above omega_s = 1", slow["jdot_acc"] > 0 > fast["jdot_acc"] and slow["fastness"] < 1 < fast["fastness"],
           f"ws={slow['fastness']:.2f}: {slow['jdot_acc']:+.1e}; ws={fast['fastness']:.2f}: {fast['jdot_acc']:+.1e}")
    report("torque vanishes at r_m = r_co (spin equilibrium)", abs(eq["fastness"] - 1) < 1e-9 and abs(eq["jdot_acc"]) < 1e-6 * slow["jdot_acc"], f"|ws-1|={abs(eq['fastness'] - 1):.1e}")
    report("loss torque = L_sd / Omega (dipole braking)", abs(tq["jdot_loss"] * ref.omega / tq["L_sd"] - 1) < 1e-12, f"Jdot_loss={tq['jdot_loss']:.3e}")
    B, R, M = cfg.B_SURFACE_OBS, cfg.NS_RADIUS, cfg.NS_MASS_OBS
    r1, r2 = alfven_radius(B, R, M, mdot), alfven_radius(B, R, M, 10 * mdot)
    report("Alfven radius scales as Mdot^(-2/7); r_m never inside the star", abs(r1 / r2 / 10 ** (2 / 7) - 1) < 1e-9 and magnetospheric_radius(B, R, M, 1e6 * mdot) == R,
           f"r_A={r1 / 1e3:.1f} km at 0.5 Mdot_max")
    L = ref.torques(0.0)["L_sd"]
    report("L_sd within a factor 3 of the observed 4.4e34 erg/s", 1 / 3 < L / cfg.SPINDOWN_LUMINOSITY_OBS < 3, f"L_sd={L * 1e7:.2e} erg/s")


def check_disk():
    section("ACCRETION DISK   32 Keplerian rings, viscous inflow, self-consistent truncation")
    m = steady_model(1.0)
    d = m.disk
    om = d.omega(m.ns.mass)
    report("ring rotation is Keplerian Omega = sqrt(GM/r^3)", np.max(np.abs(om / np.sqrt(cfg.G * m.ns.mass / d.r ** 3) - 1)) < 1e-12 and np.all(np.diff(om) < 0),
           f"Omega_in={om[0]:.2e}, Omega_out={om[-1]:.2e} rad/s")
    report("inner edge = neutron-star model r_m", abs(d.r_in / m.tq["r_m"] - 1) < 1e-9, f"r_in={d.r_in / 1e3:.1f} km")
    T = d.temperature(m.ns.mass)
    report("temperature falls outward (Shakura-Sunyaev)", T[1] > T[-1] and 1e6 < T.max() < 1e7, f"T_max={T.max():.2e} K, T_out={T[-1]:.0f} K")
    J0 = d.angular_momentum(m.ns.mass)
    m.advance(cfg.HOUR)
    dJdt = (d.angular_momentum(m.ns.mass) - J0) / cfg.HOUR
    report("angular-momentum budget closes: Jdot_in = Jdot_mag + Jdot_out + dJ/dt", abs(d.jdot_in - d.jdot_mag - d.jdot_tidal - dJdt) < 1e-9 * d.jdot_in,
           f"Jdot_in={d.jdot_in:.2e}, to magnetosphere {d.jdot_mag:.2e}")
    report("J handed to the magnetosphere = Mdot sqrt(GM r_m) (torque-model input)", abs(d.jdot_mag / (d.mdot_mag * math.sqrt(cfg.G * m.ns.mass * d.r_in)) - 1) < 1e-12, "identity")
    dd = AccretionDisk(cfg.NS_RADIUS, cfg.DISK_OUTER_RADIUS)
    total = 0.0
    for i in range(300):
        mdot = cfg.MAX_TRANSFER_RATE if i < 150 else 0.0
        total += (mdot - dd.step(mdot, 600.0, cfg.NS_MASS_OBS, lambda x: 3e4)) * 600.0
    report("ring cascade conserves mass", abs(total - dd.mass) < 1e-9 * cfg.MAX_TRANSFER_RATE * 150 * 600, f"rel err={abs(total - dd.mass) / (cfg.MAX_TRANSFER_RATE * 150 * 600):.1e}")


def check_zero_feeding():
    section("ZERO FEEDING")
    m = steady_model(1.0)
    M0, L0 = m.disk.mass, m.tq["L_acc"]
    m.set_control(0.0)
    masses, lums = [], []
    for _ in range(96):
        m.advance(cfg.HOUR)
        masses.append(m.disk.mass)
        lums.append(m.tq["L_acc"])
    report("existing disk persists, then drains gradually and monotonically", masses[0] > 0.9 * M0 and all(b < a for a, b in zip(masses, masses[1:])) and masses[-1] < 0.05 * M0,
           f"M(1h)/M0={masses[0] / M0:.3f}, M(96h)/M0={masses[-1] / M0:.1e}")
    report("accretion luminosity decreases with the disk", lums[-1] < 1e-2 * L0, f"L_acc {L0 * 1e7:.2e} -> {lums[-1] * 1e7:.2e} erg/s")
    for _ in range(200):
        m.advance(cfg.HOUR)
    f0 = m.ns.frequency
    report("accretion torque becomes exactly zero", m.tq["jdot_acc"] == 0.0 and m.tq["mdot_ns"] == 0.0 and m.state == states.PULSAR_DOMINATED, f"state={m.state}")
    m.advance(1e4 * cfg.YEAR)
    report("loss mechanism continues: dipole braking spins the star down", m.tq["jdot_loss"] > 0 and m.ns.frequency < f0, f"df={m.ns.frequency - f0:+.3e} Hz over 10 kyr")


def check_time():
    section("SIMULATION TIME   dt_sim = rate * dt_real; physics independent of warp and fps")
    T = 200.0 * cfg.DAY
    runs = []
    for rate in (cfg.DAY, 1e3 * cfg.YEAR):
        m = steady_model(0.8)
        m.warp = rate
        t0 = m.sim_time
        while m.sim_time < t0 + T - 1e-6:
            m.step(min(1.0 / 30.0, (t0 + T - m.sim_time) / m.warp))
        runs.append(m)
    a, b = runs
    report("same dM and dJ for 1 day/s and 1 kyr/s", a.ns.mass == b.ns.mass and a.ns.J == b.ns.J and abs(a.sim_time - b.sim_time) < 1e-6 * T, f"sim_time {a.clock.formatted()}")
    lo, hi = steady_model(0.9), steady_model(0.9)
    lo.warp = hi.warp = 1e3 * cfg.YEAR
    for _ in range(30):
        lo.step(1 / 30)
    for _ in range(240):
        hi.step(1 / 240)
    report("30 fps and 240 fps give the same physics", lo.ns.J == hi.ns.J and lo.ns.mass == hi.ns.mass, "bit-identical")
    c = steady_model(0.5)
    c.paused = True
    st, rt = c.sim_time, c.clock.real_elapsed
    c.step(1.0)
    report("pause advances real time but not simulated time", c.sim_time == st and c.clock.real_elapsed > rt, "clock separation")
    c.paused = False
    c.warp = 1e6 * cfg.YEAR
    c.step(1.0)
    report("kinematic cap active at extreme warp, secular dt uncapped", c.kin_capped and c.kin_dt == cfg.KINEMATIC_DT_CAP, f"kin_dt={c.kin_dt:.0f} s")


def check_beams():
    section("MAGNETIC AXIS + PULSAR BEAMS   axis from Omega, pulses from P")
    from physics.magnetosphere import SPIN_AXIS
    m = SystemModel(sim_seconds_per_real_second=1e-3)
    ax0 = m.magnetosphere.magnetic_axis
    report("magnetic axis at angle alpha to the spin axis", abs(math.acos(np.dot(ax0, SPIN_AXIS)) - m.ns.alpha) < 1e-12, f"alpha={math.degrees(m.ns.alpha):.0f} deg")
    m.step(0.4)
    ax1 = m.magnetosphere.magnetic_axis
    turned = (math.atan2(ax1[1], ax1[0]) - math.atan2(ax0[1], ax0[0])) % (2 * math.pi)
    report("axis (and beams) rotate by Omega*dt = (J/I) dt", abs(turned - (m.ns.omega * 0.4e-3) % (2 * math.pi)) < 1e-9, f"{math.degrees(turned):.3f} deg in 0.4 ms")
    m.paused = True
    m.step(1.0)
    report("paused: axis does not move", np.array_equal(ax1, m.magnetosphere.magnetic_axis), "no independent animation")
    d1, d2 = m.radiation.directions()
    report("two opposed beams on +/- m", np.allclose(d1, ax1) and np.allclose(d2, -ax1), "beam directions")
    report("pulse period = P, pulse frequency = f", m.radiation.pulse_period == m.ns.period and m.radiation.pulse_frequency == m.ns.frequency, f"f={m.ns.frequency:.2f} Hz")
    ph, I = m.radiation.profile_window(4001, 3)
    prev, nxt = np.roll(I, 1), np.roll(I, -1)
    peaks = np.where((I > prev) & (I >= nxt) & (I > 0.5 * I.max()))[0]
    report("profile over 3 P shows 3 x pulses/rotation", len(peaks) == 3 * m.radiation.pulses_per_rotation, f"{len(peaks)} peaks, {m.radiation.pulses_per_rotation} pulse/rot")
    fed = steady_model(1.0)
    report("beam intensities from the activity module (accretion channel when fed)", fed.beam_orange == fed.activity_model.accretion_intensity and fed.beam_orange > 0.8
           and fed.beam_blue == fed.activity_model.rotation_intensity, f"I_acc={fed.beam_orange:.2f}, I_rot={fed.beam_blue:.2f}")


def check_outflow():
    section("PARTICLE OUTFLOW   polar-cap wind + propeller ejection driven by activity")
    from physics.outflow import polar_cap_angle
    runs = {c: steady_model(c) for c in (0.0, 0.5, 1.0)}
    report("higher activity -> stronger polar outflow", runs[0.0].activity < runs[0.5].activity < runs[1.0].activity
           and runs[0.0].outflow.polar_index < runs[0.5].outflow.polar_index < runs[1.0].outflow.polar_index,
           f"I_polar {runs[0.0].outflow.polar_index:.2f}/{runs[0.5].outflow.polar_index:.2f}/{runs[1.0].outflow.polar_index:.2f}")
    m = runs[0.0]
    report("launch region = polar cap, theta_cap = asin sqrt(R/r_lc)", abs(m.outflow.cap_angle - polar_cap_angle(m.ns.radius, m.tq["r_lc"])) < 1e-12, f"{math.degrees(m.outflow.cap_angle):.1f} deg")
    p = steady_model(0.2)
    report("propeller: ejection index and speed sqrt(2GM/r_m) computed", p.state == states.LOW_ACCRETION and p.outflow.eq_index > 0.1
           and abs(p.outflow.eq_speed - math.sqrt(2 * cfg.G * p.ns.mass / p.tq["r_m"])) < 1e-6, f"I_eq={p.outflow.eq_index:.2f}")


def check_states():
    section("STATES   derived from omega_s, Mdot_disk, r_m, r_lc (config thresholds)")
    m = SystemModel()
    m.set_control(1.0)
    seq = [m.state]
    for _ in range(120):
        m.advance(cfg.HOUR)
        if m.state != seq[-1]:
            seq.append(m.state)
    up = [states.PULSAR_DOMINATED, states.LOW_ACCRETION, states.TRANSITION, states.ACCRETING]
    report("feeding: PULSAR -> LOW -> TRANSITION -> ACCRETING", seq == up, " -> ".join(x.split()[0] for x in seq))
    m.set_control(0.0)
    seq = [m.state]
    for _ in range(300):
        m.advance(cfg.HOUR)
        if m.state != seq[-1]:
            seq.append(m.state)
    report("starving: the reverse sequence", seq == up[::-1], " -> ".join(x.split()[0] for x in seq))
    a, b = steady_model(0.5), steady_model(0.5)
    b.set_control(0.9)
    b._refresh_derived()
    report("state ignores the control until physics responds", a.state == b.state, f"state={a.state.split()[0]}")


def check_feed_chain():
    section("FEED THE BEAST   control -> disk -> M, J -> Omega -> P -> f -> beams / activity / outflow")
    m = SystemModel()
    report("control validated and clamped", m.set_control(1.7) == 1.0 and m.set_control(-3) == 0.0 and m.set_control(float("nan")) == 0.0, "set_control")
    m.set_control(0.0)
    m.advance(cfg.DAY)
    f0, J0, act0 = m.ns.frequency, m.ns.J, m.activity
    m.set_control(1.0)
    for _ in range(120):
        m.advance(cfg.HOUR)
    tq = m.tq
    report("1. supply -> disk -> Mdot_ns > 0, L_acc > 0, ACCRETING", tq["mdot_ns"] > 0.9 * m.mdot_in and m.state == states.ACCRETING, f"Mdot_ns/supply={tq['mdot_ns'] / m.mdot_in:.3f}")
    report("2. accretion torque exceeds the loss torque", tq["jdot_acc"] > tq["jdot_loss"] > 0, f"{tq['jdot_acc']:.2e} > {tq['jdot_loss']:.2e}")
    m.advance(10 * cfg.YEAR)
    report("3. J and f increase (spin-up); pulse frequency = f", m.ns.J > J0 and m.ns.frequency > f0 and m.radiation.pulse_frequency == m.ns.frequency, f"df={m.ns.frequency - f0:+.3e} Hz / 10 yr")
    report("4. activity, beam intensity and outflow rise", m.activity > act0 and m.beam_orange > 0.8 and m.outflow.polar_index > 0.5, f"activity {act0:.2f} -> {m.activity:.2f}")
    a, b = SystemModel(), SystemModel()
    a.set_control(0.0)
    b.set_control(1.0)
    report("control alone changes nothing until physics runs", a.ns.J == b.ns.J and a.beam_blue == b.beam_blue and a.activity == b.activity and a.state == b.state, "only the supply rate differs")
    import re
    import pathlib
    offenders = [p.name for p in pathlib.Path("render").glob("*.py") if re.search(r"model\.control\b", p.read_text(encoding="utf-8")) and p.name != "hud.py"]
    report("no render module reads the control directly", not offenders, f"offenders={offenders}")


def check_stability():
    section("NUMERICAL STABILITY")
    m = SystemModel()
    m.set_control(1.0)
    ok, masses = True, []
    for _ in range(1500):
        m.advance(1e6 * cfg.YEAR / 30.0)
        t = m.telemetry()
        ok &= all(math.isfinite(t[k]) for k in ("mass_msun", "freq_hz", "period_ms", "J", "E_rot_J", "B_gauss", "L_acc_W", "L_sd_W", "disk_mass_kg")) and m.ns.J >= 0
        masses.append(m.ns.mass)
    report("all telemetry finite over 50 Myr at 100 %; mass monotonic; spin bounded", ok and all(b >= a for a, b in zip(masses, masses[1:])) and m.ns.omega < 1.1 * m.ns.breakup_omega,
           f"M={masses[-1] / cfg.M_SUN:.4f} Msun, f/f_breakup={m.ns.omega / m.ns.breakup_omega:.3f}")
    z = NeutronStar(J=1.0)
    z.J = 0.0
    tq = z.torques(cfg.MAX_TRANSFER_RATE)
    report("non-rotating star handled (P=inf, f=0, torques finite)", math.isinf(z.period) and z.frequency == 0.0 and math.isfinite(tq["jdot_acc"]) and tq["jdot_loss"] == 0.0, "safeguards")


def check_provenance():
    section("PROVENANCE   observed vs computed vs simplified")
    from physics import provenance as pv
    keys = set(steady_model(0.5).telemetry().keys())
    unclassified = sorted(k for k in keys if pv.classify(k) is None)
    stale = sorted(k for k in pv.COMPUTED_KEYS if k not in keys)
    report("every telemetry key classified; no stale registry keys", not unclassified and not stale, f"unclassified={unclassified}, stale={stale}")
    obs = pv.observed_inputs()
    report("every OBSERVED input has a citation", all("[" in e[5] for e in obs) and len(obs) >= 4, f"{len(obs)} observed entries")
    conv = {"NS_SPIN_PERIOD_OBS": 1e3, "NS_MASS_OBS": 1 / cfg.M_SUN, "SPINDOWN_LUMINOSITY_OBS": 1e7, "B_SURFACE_OBS": 1 / cfg.GAUSS, "NS_RADIUS": 1e-3,
            "DISK_VISCOUS_TIME": 1 / cfg.DAY, "DISK_OUTER_RADIUS": 1e-3, "MAX_TRANSFER_RATE": cfg.YEAR / cfg.M_SUN, "FIELD_BURIAL_MASS": 1 / cfg.M_SUN, "TOV_MASS_LIMIT": 1 / cfg.M_SUN}
    mism = [a for n, t, a, v, u, s, note in pv.INPUTS if a and v is not None and abs(getattr(cfg, a) * conv.get(a, 1.0) - v) > 1e-9 * max(1.0, abs(v))]
    report("registry values equal config.py values", not mism, f"mismatch={mism}")
    readme = open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else ""
    report("README table matches the registry", all(r in readme for r in pv.format_table(concise=True).splitlines()[:8]), "first rows verbatim")


def check_kernels():
    section("CAUSALITY AUDIT   visual buffers traced to physics (CPU kernels, f32)")
    import taichi as ti
    ti.init(arch=ti.cpu, random_seed=3)
    from physics.magnetosphere import magnetic_basis
    from render.beams import Beams
    from render.disk_particles import DiskParticles
    from render.wind import Wind
    from render.fieldlines import FieldLines
    from render.scene_map import to_scene, to_physical
    rng = np.random.default_rng(3)
    m = steady_model(1.0)
    m.kin_dt = 1.0 / 30.0
    f_prev = m.ns.frequency
    m.advance(100 * cfg.YEAR)
    fd_num = (m.ns.frequency - f_prev) / (100 * cfg.YEAR)
    report("telemetry f-dot = finite-difference df/dt (includes dI/dt)", abs(m.telemetry()["fdot_hz_s"] / fd_num - 1) < 2e-3, f"{m.telemetry()['fdot_hz_s']:.4e} vs {fd_num:.4e} Hz/s")
    basis = magnetic_basis(m.ns.spin_phase, m.ns.alpha)
    beams = Beams(50, rng)
    beams.update(m, basis)
    core = beams.core_verts.to_numpy().astype(np.float64)
    report("beam geometry axis == magnetic axis m(spin_phase, alpha)", np.allclose(core[1] / np.linalg.norm(core[1]), basis[2], atol=1e-6), "wire/core from the basis")
    report("beam cones end at the light cylinder", abs(np.linalg.norm(core, axis=1).max() / to_scene(m.tq["r_lc"]) - 1) < 1e-5, "length = s(r_lc)")
    disk = DiskParticles(32 * 40, rng)
    disk.update(m)
    disk.phi.fill(0.0)
    r0 = disk.r.to_numpy().astype(np.float64)
    m.kin_dt = 1e-4
    disk.update(m)
    keep = disk.r.to_numpy() <= r0
    dphi = disk.phi.to_numpy().astype(np.float64)
    expect = np.sqrt(cfg.G * m.ns.mass / r0 ** 3) * 1e-4
    err = np.max(np.abs(dphi[keep] - expect[keep]) / expect[keep])
    report("disk kernel: dphi = sqrt(GM/r^3) dt (Keplerian)", err < 2e-3, f"max rel err={err:.1e}")
    shown = (disk.radius.to_numpy() > 0).reshape(32, 40).sum(axis=1)
    want = np.floor(40 * m.disk.fill_fraction(cfg.MAX_TRANSFER_RATE) + 1e-9).astype(int)
    report("disk particles shown per ring == 40 * m_k/m_k,steady", np.max(np.abs(shown - want)) <= 1, f"max |shown-want|={np.max(np.abs(shown - want))}")
    m.kin_dt = 1.0 / 30.0
    wind = Wind(400, rng)
    for _ in range(60):
        wind.update(m, basis)
    sp, se = wind.visible_counts()
    wp, we = m.outflow.visual_counts(200, 200)
    report("outflow particles shown track the activity-scaled budget", sp <= wp and se <= we and sp >= 0.6 * wp - 2 and se >= 0.6 * we - 2, f"polar {sp}/{wp}, equatorial {se}/{we}")
    q = steady_model(0.0)
    fl = FieldLines(cfg.N_FIELD_LINES, cfg.FIELD_LINE_SEGMENTS)
    fl.update(q)
    nq = np.linalg.norm(fl.verts.to_numpy().astype(np.float64), axis=1)
    rq = max(to_physical(x) for x in nq[nq < 1e3])
    report("closed field lines reach the light cylinder when starved", abs(rq / q.tq["r_lc"] - 1) < 2e-3, f"{rq / 1e3:.1f} km vs r_lc {q.tq['r_lc'] / 1e3:.1f} km")
    fl.update(m)
    report("no closed field lines outside the star when the disk reaches the surface", not (np.linalg.norm(fl.verts.to_numpy(), axis=1) < 1e3).any(), f"r_m={m.tq['r_m'] / 1e3:.1f} km")
    rr = np.array([5e3, 1.2e4, 5e4, 1e6, 1e9])
    report("scene map round trip", max(abs(to_physical(to_scene(r)) - r) / r for r in rr) < 1e-12, "log map")
    report("no visual reads wall-clock time or frame counters", not any(("perf_counter" in open(os.path.join("render", f), encoding="utf-8").read()
           or "time.time" in open(os.path.join("render", f), encoding="utf-8").read()) for f in os.listdir("render") if f.endswith(".py")), "source scan")


def run_checks(force_cpu=False):
    print("FEED THE BEAST - scientific validation of the core equations")
    print("=" * 78)
    for fn in (check_rotation, check_mass, check_angular_momentum, check_disk, check_zero_feeding, check_time,
               check_beams, check_outflow, check_states, check_feed_chain, check_stability, check_provenance, check_kernels):
        fn()
    n_ok, n = sum(RESULTS), len(RESULTS)
    print("\n" + "=" * 78)
    print(f"{'ALL CHECKS PASSED' if n_ok == n else 'CHECKS FAILED'}: {n_ok}/{n}")
    return n_ok == n


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_checks() else 1)
