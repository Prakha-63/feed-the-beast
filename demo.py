"""
Scripted demo / recording mode:  python sim.py --demo [--demo-duration 60]

Drives ONLY the existing accretion control (SystemModel.set_control) on a
fixed real-time schedule; everything else - disk, torque, spin, beams,
telemetry - follows through the normal physics.  The camera stays on a
preset with a slow, steady orbit (visual only).  The window closes when the
sequence ends, so a screen recorder can capture start to finish.

Sequence (fractions of the total duration):
    0.00-0.20  50 %   stable baseline (disk builds, state ACCRETING)
    0.20-0.45  100 %  heavy feeding: disk brightens, torque and f-dot rise
    0.45-0.95  0 %    starve: disk drains, L_acc falls, beams turn blue
    0.95-1.00  0 %    final state held (PULSAR-DOMINATED, dipole braking)
"""
PHASES = [(0.00, 0.5, "BASELINE  50 %"), (0.20, 1.0, "HEAVY FEEDING  100 %"),
          (0.45, 0.0, "STARVE  0 %"), (0.95, 0.0, "FINAL STATE")]
CAMERA_PRESET = "1"
ORBIT_RATE = 0.03          # rad/s slow yaw drift (camera only)


class DemoScript:
    def __init__(self, duration):
        self.duration = max(10.0, float(duration))
        self.elapsed = 0.0
        self.label = ""
        self.done = False
        self._started = False

    def update(self, sim, real_dt):
        if not self._started:
            sim.renderer.camera.set_preset(CAMERA_PRESET)
            self._started = True
        self.elapsed += real_dt
        frac = self.elapsed / self.duration
        control, label = PHASES[0][1], PHASES[0][2]
        for start, c, text in PHASES:
            if frac >= start:
                control, label = c, text
        sim.model.set_control(control)                       # the only thing the demo touches
        sim.renderer.camera.target[0] += ORBIT_RATE * real_dt
        self.label = f"DEMO {self.elapsed:4.0f}/{self.duration:.0f} s   {label}"
        self.done = frac >= 1.0
