"""
Explicit state-transition model.

OBSERVED-INSPIRED BEHAVIOUR (what the model is *inspired by*):
    PSR J1023+0038 is a transitional millisecond pulsar.  It has been seen in
    a rotation-powered radio-pulsar state (no disk, L_X ~ 1e32 erg/s) and, since
    2013, in a sub-luminous accretion-disk state (disk present, radio pulsar
    undetectable, X-ray pulsations and rapid mode switching, L_X ~ 1e33-1e34
    erg/s).  The change happened on a timescale of weeks.  (Archibald et al.
    2009, 2015; Stappers et al. 2014; Patruno et al. 2014.)

SIMULATED CRITERIA (what this code actually does):
    The state is a pure function of computed variables - never of a UI
    control - using the thresholds in config.py:

        PULSAR_DOMINATED   Mdot_disk < ACCRETION_FLOOR   (no disk feeding the
                           magnetosphere)  or  r_m >= r_lc  (ejector: the disk
                           is held outside the light cylinder)
        LOW_ACCRETION      omega_s > STATE_PROPELLER_MIN_FASTNESS  (propeller:
                           matter reaches r_m but is mostly flung out)
        TRANSITION         STATE_ACCRETING_MAX_FASTNESS <= omega_s <=
                           STATE_PROPELLER_MIN_FASTNESS  (partial accretion,
                           torque near zero)
        ACCRETING          omega_s < STATE_ACCRETING_MAX_FASTNESS

    with  omega_s = (r_m / r_co)^(3/2),  r_m = xi r_A(Mdot_disk),  r_co =
    (G M / Omega^2)^(1/3),  r_lc = c / Omega.

    Because Mdot_disk lags the user's control through the stream transit and
    the disk's viscous inflow, transitions happen on the model's own
    timescales (hours-days at DISK_VISCOUS_TIME = 1 day), not instantly.

WHAT THIS IS NOT: a reproduction of PSR J1023+0038.  The real transitions are
probably driven by disk instabilities and irradiation that are not modelled;
the thresholds above are a simplified, explicit mapping from the fastness
parameter to qualitative regimes.  No hysteresis is modelled.

Feeding label (from the user's control only - a UI descriptor, not a state):
    STARVING / LOW / NORMAL / HEAVY
Stability flag:
    EXTREME when M / M_TOV > 0.9 or Omega / Omega_breakup > 0.9
"""
import math

import config as cfg

# --- state identifiers (display strings) --------------------------------------
PULSAR_DOMINATED = "PULSAR-DOMINATED"
LOW_ACCRETION = "LOW ACCRETION (PROPELLER)"
TRANSITION = "TRANSITION"
ACCRETING = "ACCRETING"
ALL_STATES = (PULSAR_DOMINATED, LOW_ACCRETION, TRANSITION, ACCRETING)

# short aliases kept for existing code
PULSAR = PULSAR_DOMINATED
LOW = LOW_ACCRETION

STATE_COLORS = {
    PULSAR_DOMINATED: (0.45, 0.65, 1.0),
    LOW_ACCRETION: (0.9, 0.55, 0.9),
    TRANSITION: (1.0, 0.85, 0.4),
    ACCRETING: (1.0, 0.5, 0.2),
}

# ordering used to describe a transition as "toward accretion" or "toward pulsar"
STATE_RANK = {PULSAR_DOMINATED: 0, LOW_ACCRETION: 1, TRANSITION: 2, ACCRETING: 3}


def classify(mdot_disk, r_m, r_lc, fastness):
    """Simulated criteria (see module docstring)."""
    if mdot_disk < cfg.ACCRETION_FLOOR or r_m >= r_lc:
        return PULSAR_DOMINATED
    if fastness > cfg.STATE_PROPELLER_MIN_FASTNESS:
        return LOW_ACCRETION
    if fastness >= cfg.STATE_ACCRETING_MAX_FASTNESS:
        return TRANSITION
    return ACCRETING


def criteria_text(mdot_disk, r_m, r_lc, fastness):
    """Human-readable evaluation of the criteria (telemetry)."""
    if mdot_disk < cfg.ACCRETION_FLOOR:
        return "Mdot_disk < floor"
    if r_m >= r_lc:
        return "r_m >= r_lc (ejector)"
    if fastness > cfg.STATE_PROPELLER_MIN_FASTNESS:
        return f"omega_s = {fastness:.2f} > {cfg.STATE_PROPELLER_MIN_FASTNESS}"
    if fastness >= cfg.STATE_ACCRETING_MAX_FASTNESS:
        return f"{cfg.STATE_ACCRETING_MAX_FASTNESS} <= omega_s = {fastness:.2f} <= {cfg.STATE_PROPELLER_MIN_FASTNESS}"
    return f"omega_s = {fastness:.2f} < {cfg.STATE_ACCRETING_MAX_FASTNESS}"


class StateMachine:
    """Tracks the current state, time spent in it and the transition history.
    Fed only with physical variables."""

    def __init__(self):
        self.state = PULSAR_DOMINATED
        self.since = 0.0             # sim time when the current state began
        self.transitions = []        # (sim_time, from_state, to_state)
        self.n_transitions = 0
        self.criteria = ""

    def update(self, sim_time, mdot_disk, r_m, r_lc, fastness):
        new = classify(mdot_disk, r_m, r_lc, fastness)
        self.criteria = criteria_text(mdot_disk, r_m, r_lc, fastness)
        if new != self.state:
            self.transitions.append((sim_time, self.state, new))
            if len(self.transitions) > cfg.STATE_HISTORY_LENGTH:
                self.transitions.pop(0)
            self.n_transitions += 1
            self.state = new
            self.since = sim_time
        return self.state

    def time_in_state(self, sim_time):
        return sim_time - self.since

    def direction(self):
        """+1 if the last transition moved toward accretion, -1 toward pulsar, 0 if none."""
        if not self.transitions:
            return 0
        _, a, b = self.transitions[-1]
        return int(math.copysign(1, STATE_RANK[b] - STATE_RANK[a]))

    def reset(self):
        self.__init__()


def feeding_label(control):
    if control <= 0.0:
        return "STARVING"
    if control < 0.35:
        return "LOW"
    if control < 0.75:
        return "NORMAL"
    return "HEAVY"


def stability_index(mass, omega, breakup_omega):
    return max(mass / cfg.TOV_MASS_LIMIT, omega / breakup_omega)
