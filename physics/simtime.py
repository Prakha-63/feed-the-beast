"""
Simulation-time system.

Two clocks are kept strictly apart:

    REAL TIME        wall-clock seconds, only used to decide how much
                     simulated time a frame covers.  Nothing physical depends
                     on it.
    SIMULATION TIME  seconds of the modelled system.  Every physical update
                     (mass, angular momentum, orbit, disk) is a function of
                     simulated dt only.

    dt_sim = simulation_seconds_per_real_second * dt_real

The conversion factor ("time warp") is a plain configurable float; the
preset table in config.TIME_WARPS is a convenience for stepping through
useful values.  Changing the factor - or the frame rate - changes how fast
the modelled system evolves for the viewer, never the relationships being
simulated (validated in checks.py and tests/test_simtime.py).

Kinematic slice: visual phases (orbit position, spin phase, disk azimuth)
advance by  kin_dt = min(dt_sim, KINEMATIC_DT_CAP)  so the picture stays
readable at 1 s = 1 Myr.  Secular physics always receives the full dt_sim.
"""
import math

import config as cfg


def format_duration(t):
    """Human-readable simulated duration, e.g. '+ 3 d 04:12:07.250'."""
    if not math.isfinite(t):
        return "inf"
    sign = "-" if t < 0 else "+"
    t = abs(t)
    if t >= cfg.YEAR:
        yr = t / cfg.YEAR
        if yr >= 1e6:
            return f"{sign} {yr / 1e6:.3f} Myr"
        if yr >= 1e3:
            return f"{sign} {yr / 1e3:.3f} kyr"
        return f"{sign} {yr:.3f} yr"
    days, rem = divmod(t, cfg.DAY)
    hours, rem = divmod(rem, cfg.HOUR)
    minutes, seconds = divmod(rem, 60.0)
    if days >= 1:
        return f"{sign} {int(days)} d {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    if hours >= 1:
        return f"{sign} {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    if minutes >= 1:
        return f"{sign} {int(minutes):02d}:{seconds:06.3f}"
    if t >= 1.0:
        return f"{sign} {seconds:.3f} s"
    return f"{sign} {t * 1e3:.3f} ms"


def format_rate(sim_per_real):
    """'1 s = 1 hour'-style label for a warp factor."""
    for value, label in cfg.TIME_WARPS:
        if math.isclose(value, sim_per_real, rel_tol=1e-9):
            return label
    return f"1 s = {format_duration(sim_per_real)[2:]}"


class SimulationClock:
    def __init__(self, sim_seconds_per_real_second=None, kinematic_cap=cfg.KINEMATIC_DT_CAP):
        self.kinematic_cap = kinematic_cap
        self.paused = False
        self.real_elapsed = 0.0          # wall-clock seconds since (re)start, incl. paused
        self.real_running = 0.0          # wall-clock seconds while not paused
        self.sim_time = 0.0              # simulated seconds since (re)start
        self.last_dt = 0.0               # simulated seconds of the last tick
        self.kin_dt = 0.0                # kinematic slice of the last tick
        self.kin_capped = False
        self.ticks = 0
        if sim_seconds_per_real_second is None:
            self.warp_index = cfg.DEFAULT_WARP_INDEX
            self._rate = cfg.TIME_WARPS[self.warp_index][0]
        else:
            self.rate = sim_seconds_per_real_second

    # ------------------------------------------------------------- rate
    @property
    def rate(self):
        """simulation_seconds_per_real_second (float, > 0)."""
        return self._rate

    @rate.setter
    def rate(self, value):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("simulation_seconds_per_real_second must be finite and > 0")
        self._rate = float(value)
        # snap the preset index to the nearest table entry (for UI stepping)
        self.warp_index = min(range(len(cfg.TIME_WARPS)),
                              key=lambda i: abs(math.log(cfg.TIME_WARPS[i][0]) - math.log(value)))

    @property
    def rate_label(self):
        return format_rate(self._rate)

    def step_preset(self, delta):
        """Move through config.TIME_WARPS by delta entries."""
        self.warp_index = max(0, min(len(cfg.TIME_WARPS) - 1, self.warp_index + delta))
        self._rate = cfg.TIME_WARPS[self.warp_index][0]

    # ------------------------------------------------------------- tick
    def tick(self, real_dt):
        """Consume real_dt wall-clock seconds; return the simulated dt to
        integrate (0 while paused or for invalid input)."""
        if not math.isfinite(real_dt) or real_dt <= 0.0:
            return 0.0
        self.real_elapsed += real_dt
        if self.paused:
            self.last_dt = self.kin_dt = 0.0
            return 0.0
        self.real_running += real_dt
        dt = self._rate * real_dt
        self.advance(dt)
        return dt

    def advance(self, dt):
        """Book-keeping for a direct simulated-time advance (no real clock)."""
        if not math.isfinite(dt) or dt <= 0.0:
            return 0.0
        self.last_dt = dt
        self.kin_dt = min(dt, self.kinematic_cap)
        self.kin_capped = self.kin_dt < dt
        self.sim_time += dt
        self.ticks += 1
        return dt

    def reset(self):
        self.real_elapsed = self.real_running = self.sim_time = 0.0
        self.last_dt = self.kin_dt = 0.0
        self.kin_capped = False
        self.ticks = 0

    # ------------------------------------------------------------- readouts
    @property
    def elapsed_simulated_seconds(self):
        return self.sim_time

    def breakdown(self):
        """Simulated time split into whole days / hours / minutes / seconds."""
        t = self.sim_time
        days, rem = divmod(t, cfg.DAY)
        hours, rem = divmod(rem, cfg.HOUR)
        minutes, seconds = divmod(rem, 60.0)
        return {"days": int(days), "hours": int(hours), "minutes": int(minutes), "seconds": seconds,
                "years": t / cfg.YEAR, "total_seconds": t}

    def formatted(self):
        return format_duration(self.sim_time)

    def snapshot(self):
        return {
            "sim_time_s": self.sim_time,
            "sim_time_formatted": self.formatted(),
            "sim_time_breakdown": self.breakdown(),
            "real_elapsed_s": self.real_elapsed,
            "real_running_s": self.real_running,
            "sim_seconds_per_real_second": self._rate,
            "rate_label": self.rate_label,
            "last_dt_s": self.last_dt,
            "kin_dt_s": self.kin_dt,
            "kin_capped": self.kin_capped,
            "paused": self.paused,
        }
