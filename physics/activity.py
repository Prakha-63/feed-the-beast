"""
Modelled activity: one dimensionless variable and its channel components,
all computed from luminosities / rates of the physical model.  This module is
the single place where physical power is mapped to the [0, 1] indices that
drive radiation intensity, particle outflow, disk visuals and telemetry.

    log_index(P)   = clip( log10(P / L_LOW) / log10(L_HIGH / L_LOW), 0, 1 )
                     with L_LOW = 1e26 W, L_HIGH = 1.3e31 W (~Eddington)

    activity       = log_index(L_acc + L_sd)          total modelled power
    accretion_act  = log_index(L_acc)
    rotation_act   = log_index(L_sd)
    disk_act       = log_index(L_disk)                disk emission
    polar_power    = L_sd + JET_FRACTION * L_acc      (physics/outflow.py)

Channel intensities used by the beams (physics/radiation.py):
    rotation_intensity  = min(1, L_sd / L_sd,obs) * quench
                          quench = 1 in PULSAR_DOMINATED, ACTIVITY_QUENCH_WITH_DISK
                          otherwise (radio pulsar hidden by the disk: observed-
                          inspired behaviour, simplified as a constant factor)
    accretion_intensity = sqrt( L_acc / L_acc,max ),  L_acc,max = G M Mdot_max / R

Everything here is a deterministic function of the physics; there is no
independent "activity" control.
"""
import math

import config as cfg
from physics import states

L_LOW, L_HIGH = cfg.ACTIVITY_L_LOW, cfg.ACTIVITY_L_HIGH


def log_index(P, lo=L_LOW, hi=L_HIGH):
    if not math.isfinite(P) or P <= 0.0:
        return 0.0
    return min(1.0, max(0.0, math.log10(P / lo) / math.log10(hi / lo)))


def rotation_intensity(L_sd, state):
    quench = 1.0 if state == states.PULSAR_DOMINATED else cfg.ACTIVITY_QUENCH_WITH_DISK
    return min(1.0, max(0.0, L_sd) / cfg.SPINDOWN_LUMINOSITY_OBS) * quench


def accretion_intensity(L_acc, M, R):
    L_max = cfg.G * M * cfg.MAX_TRANSFER_RATE / R
    return min(1.0, max(0.0, L_acc) / L_max) ** 0.5


class Activity:
    def __init__(self):
        self.total = 0.0
        self.accretion = 0.0
        self.rotation = 0.0
        self.disk = 0.0
        self.L_total = 0.0
        self.rotation_intensity = 0.0
        self.accretion_intensity = 0.0

    def update(self, tq, L_disk, state, M, R):
        self.L_total = tq["L_acc"] + tq["L_sd"]
        self.total = log_index(self.L_total)
        self.accretion = log_index(tq["L_acc"])
        self.rotation = log_index(tq["L_sd"])
        self.disk = log_index(L_disk)
        self.rotation_intensity = rotation_intensity(tq["L_sd"], state)
        self.accretion_intensity = accretion_intensity(tq["L_acc"], M, R)
        return self.total

    def telemetry(self):
        return {
            "activity": self.total,
            "activity_accretion": self.accretion,
            "activity_rotation": self.rotation,
            "activity_disk": self.disk,
            "L_total_W": self.L_total,
            "beam_I_rotation": self.rotation_intensity,
            "beam_I_accretion": self.accretion_intensity,
        }
