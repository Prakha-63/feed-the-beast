"""
HUD: accretion control, live telemetry and the pulse profile.

Every number shown comes from SystemModel.telemetry(); nothing is fabricated.
The pulse profile is the modelled observed intensity versus rotation phase
for a fixed observer inclination, computed from the beam geometry:
    I(phi) = (I_blue + I_orange) * sum_{+/-} exp(-(angle(+/-m(phi), n)/w)^2)
The x-axis spans the last three rotations (3 P), so its physical width in ms
changes as P changes.
"""
import numpy as np
import taichi as ti

import config as cfg
from render.telemetry_text import telemetry_lines, control_lines

N_PROFILE = 192
N_ROTATIONS = 3


class Hud:
    def __init__(self):
        self.prof_verts = ti.Vector.field(2, ti.f32, 2 * (N_PROFILE - 1) + 4)     # profile segments + 2 axis segments
        self.prof_color = (0.3, 0.3, 0.35)

    # ---------------------------------------------------------------- profile
    def _pulse_profile(self, model):
        _, I = model.radiation.profile_window(N_PROFILE, N_ROTATIONS)   # derived from P and the beam geometry
        k = np.arange(N_PROFILE)
        x = 0.02 + 0.34 * k / (N_PROFILE - 1)
        y = 0.06 + 0.11 * I
        pts = np.stack([x, y], axis=1).astype(np.float32)
        seg = np.empty((2 * (N_PROFILE - 1) + 4, 2), dtype=np.float32)
        seg[0:-4:2] = pts[:-1]
        seg[1:-4:2] = pts[1:]
        seg[-4:] = [[0.02, 0.055], [0.36, 0.055], [0.02, 0.055], [0.02, 0.17]]
        self.prof_verts.from_numpy(seg)
        amp = model.beam_blue + model.beam_orange
        blue = np.array([0.45, 0.65, 1.0]) * model.beam_blue
        orange = np.array([1.0, 0.55, 0.2]) * model.beam_orange
        c = np.clip(blue + orange, 0, 1) if amp > 0.02 else np.array([0.3, 0.3, 0.35])
        self.prof_color = tuple(float(x) for x in c)

    # ------------------------------------------------------------------- draw
    def draw(self, window, canvas, sim, camera_name):
        model = sim.model
        t = model.telemetry()
        # Pulse profile (2D canvas lines) is drawn in every mode.
        self._pulse_profile(model)
        canvas.lines(self.prof_verts, 0.003, color=self.prof_color)
        if not sim.run.show_window:
            # Headless (screenshot / smoke-test) mode: GGUI's imgui pass without
            # window.show() prevents the frame buffer from being cleared, so
            # frames would accumulate.  The text panels are skipped here.
            return
        gui = window.get_gui()

        # --- control panel ---------------------------------------------------
        with gui.sub_window("FEED THE BEAST  -  " + cfg.REF_SYSTEM_NAME, 0.01, 0.01, 0.31, 0.33):
            model.set_control(gui.slider_float("ACCRETION RATE  0 % .. 100 %", model.control, 0.0, 1.0))
            lines = control_lines(t, model.warp_label, camera_name, sim.backend, sim.fps)
            gui.text(lines[0])
            if gui.button("PAUSE" if not model.paused else "RESUME"):
                sim.toggle_pause()
            if gui.button("RESET"):
                sim.reset()
            if gui.button("RESTART"):
                sim.restart()
            for line in lines[1:]:
                gui.text(line)

        # --- telemetry (single panel; every value from model.telemetry()) ----
        with gui.sub_window("TELEMETRY", 0.58, 0.01, 0.41, 0.52):
            for text, color in telemetry_lines(t, model.warp_label, model.kin_capped):
                if color is None:
                    gui.text(text)
                else:
                    gui.text(text, color=color)

        # --- pulse profile label ---------------------------------------------
        with gui.sub_window(f"PULSE PROFILE (modelled, observer i={t['observer_inclination_deg']:.0f} deg)", 0.01, 0.80, 0.29, 0.06):
            gui.text(f"window = {N_ROTATIONS} P = {N_ROTATIONS * t['pulse_period_ms']:.4f} ms   f = {t['pulse_frequency_hz']:.2f} Hz"
                     f"   {t['pulses_per_rotation']} pulse/rot")
