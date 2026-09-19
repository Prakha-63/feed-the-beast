"""
SystemModel: the connected causal system.

    control (0..1)  ->  Mdot_in = control^3 * MAX_TRANSFER_RATE   (supply at the disk edge)
                    ->  ring disk (Keplerian annuli, viscous inflow) -> Mdot_disk at r_m
                    ->  r_m, r_co, fastness                -> f_acc, torques
                    ->  dM/dt = f_acc * Mdot_disk
                        dJ/dt = Jdot_acc - Jdot_md
                    ->  Omega = J/I, P = 2pi/Omega, f = 1/P, E_rot
                    ->  luminosities, state, activity      -> visualization

Time: all physics advances in *simulated* seconds supplied by
physics.simtime.SimulationClock (dt_sim = rate * dt_real).  See that module
for the real/simulated separation and the kinematic slice used for visual
phases.
"""
import math

import config as cfg
from physics.neutron_star import NeutronStar, magnetospheric_radius
from physics.disk import AccretionDisk
from physics import states
from physics.simtime import SimulationClock
from physics.magnetosphere import Magnetosphere
from physics.radiation import PulsarRadiation
from physics.outflow import Outflow
from physics.activity import Activity


class SystemModel:
    def __init__(self, substeps=cfg.SECULAR_SUBSTEPS, sim_seconds_per_real_second=None):
        self.substeps = max(1, int(substeps))   # RK2 substeps per advance()
        self.clock = SimulationClock(sim_seconds_per_real_second)
        self.ns = NeutronStar()
        self.disk = AccretionDisk(self.ns.radius, cfg.DISK_OUTER_RADIUS)
        self.magnetosphere = Magnetosphere(self.ns)
        self.radiation = PulsarRadiation(self.ns)
        self.outflow = Outflow(self.ns)
        self.state_machine = states.StateMachine()
        self.activity_model = Activity()
        self.control = 0.5          # accretion-rate control, 0..1
        self.mdot_disk = 0.0        # kg/s delivered to the magnetosphere (last step)
        self.tq = self.ns.torques(0.0)
        self.state = states.PULSAR_DOMINATED
        self._refresh_derived()

    # ------------------------------------------------------------------ time
    # Thin delegates so existing callers keep working; the clock is the owner.
    @property
    def warp(self):
        """simulation_seconds_per_real_second."""
        return self.clock.rate

    @warp.setter
    def warp(self, value):
        self.clock.rate = value

    @property
    def warp_label(self):
        return self.clock.rate_label

    @property
    def warp_index(self):
        return self.clock.warp_index

    @warp_index.setter
    def warp_index(self, i):
        self.clock.step_preset(i - self.clock.warp_index)

    def change_warp(self, delta):
        self.clock.step_preset(delta)

    @property
    def paused(self):
        return self.clock.paused

    @paused.setter
    def paused(self, v):
        self.clock.paused = bool(v)

    @property
    def sim_time(self):
        return self.clock.sim_time

    @property
    def kin_dt(self):
        return self.clock.kin_dt

    @kin_dt.setter
    def kin_dt(self, v):          # used by kernel checks to force a slice
        self.clock.kin_dt = v

    @property
    def kin_capped(self):
        return self.clock.kin_capped

    # ------------------------------------------------------------------ control
    @property
    def mdot_in(self):
        """Mass supply rate at the disk edge.  Cubic mapping so that the
        slider spans ~4 decades (1e-12 .. 1e-9 Msun/yr) with the
        propeller/accretion boundary near the middle.  Documented in README."""
        return cfg.MAX_TRANSFER_RATE * min(1.0, max(0.0, self.control)) ** 3

    # ------------------------------------------------------------- initial state
    # RESET restores exactly this (config.py reference inputs):
    INITIAL_STATE_DOC = (
        "neutron star: M = NS_MASS_OBS, R = NS_RADIUS, B = B_SURFACE_OBS, alpha = MAGNETIC_MISALIGNMENT, "
        "J = I * 2 pi / NS_SPIN_PERIOD_OBS, spin phase 0, accreted mass 0; "
        "disk empty; simulation time 0; state PULSAR_DOMINATED, no transitions.")

    def initial_state(self):
        """Snapshot of the documented initial physical state (for RESET verification)."""
        return {
            "mass": cfg.NS_MASS_OBS, "J": cfg.MOMENT_OF_INERTIA_PREFACTOR * cfg.NS_MASS_OBS * cfg.NS_RADIUS ** 2
            * 2.0 * math.pi / cfg.NS_SPIN_PERIOD_OBS, "accreted_mass": 0.0, "spin_phase": 0.0,
            "disk_mass": 0.0, "sim_time": 0.0,
            "state": states.PULSAR_DOMINATED, "transitions": 0,
        }

    def is_initial(self):
        s = self.initial_state()
        return (self.ns.mass == s["mass"] and self.ns.J == s["J"] and self.ns.accreted_mass == 0.0 and self.ns.spin_phase == 0.0
                and self.disk.mass == 0.0 and self.sim_time == 0.0
                and self.state == states.PULSAR_DOMINATED and self.state_machine.n_transitions == 0)

    def set_control(self, value):
        """The single user input: accretion-rate control in [0, 1].  It sets the
        mass supply rate at the disk edge (cubic mapping) and nothing else; every other
        quantity follows through the physics."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.control
        if not math.isfinite(v):
            return self.control
        self.control = min(1.0, max(0.0, v))
        return self.control

    def reset(self):
        rate = self.clock.rate
        self.__init__(substeps=self.substeps, sim_seconds_per_real_second=rate)

    # ------------------------------------------------------------------- step
    def step(self, real_dt):
        """Frame entry point: convert wall-clock dt to simulated dt via the
        clock, then integrate.  Physics never sees real_dt."""
        dt = self.clock.tick(real_dt)
        if dt > 0.0:
            self._integrate(dt)

    def advance(self, dt):
        """Advance the physical system by dt simulated seconds directly
        (tests, validation, scripted runs)."""
        if self.clock.advance(dt) > 0.0:
            self._integrate(dt)

    def _integrate(self, dt):
        # 1. control -> mass supply at the disk edge -> ring disk -> rate handed
        #    to the magnetosphere at the self-consistent truncation radius
        ns = self.ns
        self.mdot_disk = self.disk.step(self.mdot_in, dt, ns.mass,
                                        lambda mdot: magnetospheric_radius(ns.B, ns.radius, ns.mass, mdot))

        # 3. neutron-star mass and angular momentum, RK2 midpoint substeps
        n = self.substeps
        h = dt / n
        for _ in range(n):
            tq0 = self.ns.torques(self.mdot_disk)
            dM0, dJ0 = tq0["mdot_ns"], tq0["jdot_acc"] - tq0["jdot_loss"]
            trial = NeutronStar(mass=self.ns.mass + 0.5 * h * dM0, radius=self.ns.radius,
                                B0=self.ns.B0, alpha=self.ns.alpha, J=self.ns.J + 0.5 * h * dJ0,
                                accreted_mass=self.ns.accreted_mass + 0.5 * h * dM0)
            tqm = trial.torques(self.mdot_disk)
            self.ns.apply(h * tqm["mdot_ns"], h * (tqm["jdot_acc"] - tqm["jdot_loss"]))
        self.tq = self.ns.torques(self.mdot_disk)

        # 4. kinematic slice (set by the clock): visual phases at physical rates
        self.ns.spin_phase = (self.ns.spin_phase + self.ns.omega * self.kin_dt) % (2.0 * math.pi)

        self._refresh_derived()

    # --------------------------------------------------------------- derived
    def _refresh_derived(self):
        tq = self.tq
        # state: pure function of computed variables (physics/states.py)
        self.state = self.state_machine.update(self.sim_time, self.mdot_disk, tq["r_m"], tq["r_lc"], tq["fastness"])
        self.feeding = states.feeding_label(self.control)
        self.stability = states.stability_index(self.ns.mass, self.ns.omega, self.ns.breakup_omega)
        self.extreme = self.stability > 0.9
        # modelled activity (physics/activity.py) drives radiation, outflow, disk visuals, telemetry
        self.activity = self.activity_model.update(tq, self.disk.luminosity(self.ns.mass), self.state,
                                                   self.ns.mass, self.ns.radius)
        # visual driver indices (dimensionless, documented in README)
        self.disk_ring_fill = self.disk.fill_fraction(cfg.MAX_TRANSFER_RATE)      # per-ring visual density
        self.disk_ring_T = self.disk.temperature(self.ns.mass)
        # pulsar beams: intensities from the modelled luminosities (physics/radiation.py)
        self.radiation.update(tq, self.state)
        self.beam_blue = self.radiation.I_rot
        self.beam_orange = self.radiation.I_acc
        # particle outflow drivers (physics/outflow.py): polar cap wind + propeller ejection
        self.outflow.update(tq)
        self.r_in = self.disk.r_in                    # inner edge = r_m clamped to the disk grid

    def telemetry(self):
        """Dict of live values (SI + display units)."""
        ns, tq = self.ns, self.tq
        jdot = tq["jdot_acc"] - tq["jdot_loss"]
        # d(Omega)/dt = Jdot/I - Omega * (dI/dt)/I  with I = k M R^2  ->  (dI/dt)/I = Mdot_ns / M
        fdot = (jdot / ns.inertia - ns.omega * tq["mdot_ns"] / ns.mass) / (2.0 * math.pi)     # Hz/s
        return {
            "mass_msun": ns.mass / cfg.M_SUN,
            "accreted_mass_msun": ns.accreted_mass / cfg.M_SUN,
            "delta_f_hz": ns.frequency - 1.0 / cfg.NS_SPIN_PERIOD_OBS,
            "jdot_net": jdot,
            "control": self.control,
            "mdot_in_msun_yr": self.mdot_in * cfg.YEAR / cfg.M_SUN,
            "mdot_ns_msun_yr": tq["mdot_ns"] * cfg.YEAR / cfg.M_SUN,
            "disk_mass_kg": self.disk.mass,
            **self.disk.telemetry(self.ns.mass),
            "freq_hz": ns.frequency,
            "period_ms": ns.period * 1e3,
            "fdot_hz_s": fdot,
            "B_gauss": ns.B / cfg.GAUSS,
            **self.magnetosphere.telemetry(tq["r_m"], tq["r_lc"]),
            **self.radiation.telemetry(),
            **self.outflow.telemetry(),
            "r_m_km": tq["r_m"] / 1e3,
            "r_co_km": tq["r_co"] / 1e3,
            "r_lc_km": tq["r_lc"] / 1e3,
            "fastness": tq["fastness"],
            "L_acc_W": tq["L_acc"],
            "L_sd_W": tq["L_sd"],
            "E_rot_J": ns.rotational_energy,
            "J": ns.J,
            **self.activity_model.telemetry(),
            "state": self.state,
            "state_criteria": self.state_machine.criteria,
            "state_time_s": self.state_machine.time_in_state(self.sim_time),
            "state_transitions": self.state_machine.n_transitions,
            "state_last_transition": self.state_machine.transitions[-1] if self.state_machine.transitions else None,
            "state_direction": self.state_machine.direction(),
            "feeding": self.feeding,
            "stability": self.stability,
            "extreme": self.extreme,
            "sim_time_s": self.sim_time,
            "sim_time_formatted": self.clock.formatted(),
            "sim_time_breakdown": self.clock.breakdown(),
            "real_elapsed_s": self.clock.real_elapsed,
            "sim_seconds_per_real_second": self.clock.rate,
        }
