"""
Orbit camera - visualization only.

This module imports nothing from physics/ and never writes to the model.
The focus point is the neutron star (scene origin) plus the user's pan.

Mouse
    left-drag            orbit (yaw / pitch)
    right-drag           pan: moves the focus point in the view plane
    middle-drag          zoom (drag up = in);  Shift + left-drag also zooms
    (GGUI exposes no scroll-wheel event, so the wheel is not used)
Keyboard
    UP / W / +           zoom in            DOWN / S / -   zoom out
    1  free orbit        2  neutron-star close-up   3  disk view
    4  top view          5  side view               6  far view
    0                    re-centre the focus on the current preset's anchor
Presets set yaw, pitch and distance.  All motion is eased over a few frames.
"""
import math

import numpy as np
import taichi as ti

#            name             yaw   pitch  dist
PRESETS = {
    "1": ("FREE ORBIT",       0.6,  0.42,  38.0),
    "2": ("NS CLOSE-UP",      0.9,  0.70,  15.0),
    "3": ("DISK VIEW",        0.4,  0.55,  55.0),
    "4": ("TOP VIEW",         0.0,  1.50,  55.0),
    "5": ("SIDE VIEW",        0.0,  0.02,  45.0),
    "6": ("FAR VIEW",         0.3,  0.45,  90.0),
}
MIN_DIST, MAX_DIST = 3.0, 220.0
MAX_PITCH = 1.55
ORBIT_SENS = 4.0        # rad per full-window drag
PITCH_SENS = 3.0
ZOOM_KEY_RATE = 2.5     # e-folds per second while a zoom key is held
ZOOM_DRAG_SENS = 3.0    # e-folds per full-window vertical drag
PAN_SENS = 1.2          # fraction of the view height per full-window drag
EASE_RATE = 6.0         # 1/s


class OrbitCamera:
    def __init__(self):
        self.cam = ti.ui.Camera()
        self.name = PRESETS["1"][0]
        # targets (what input sets) and eased values (what is rendered)
        self.target = np.array(PRESETS["1"][1:4], dtype=float)      # yaw, pitch, dist
        self.yaw, self.pitch, self.dist = self.target
        self.pan_target = np.zeros(3)                               # offset added to the anchor, scene units
        self.pan = np.zeros(3)
        self.focus = np.zeros(3)                                    # eased look-at point
        self._last = {}                                             # last cursor per button
        self.position = np.array([self.dist, 0.0, 0.0])

    # ------------------------------------------------------------ presets
    def set_preset(self, key):
        if key in PRESETS:
            name, yaw, pitch, dist = PRESETS[key]
            self.name = name
            self.target[:] = (yaw, pitch, dist)
            self.pan_target[:] = 0.0

    def recentre(self):
        self.pan_target[:] = 0.0

    # ------------------------------------------------------------ input
    def _drag(self, window, button, key):
        """Return (dx, dy) cursor motion while `button` is held, else None."""
        if window.is_pressed(button):
            cur = window.get_cursor_pos()
            last = self._last.get(key)
            self._last[key] = cur
            if last is not None:
                return cur[0] - last[0], cur[1] - last[1]
            return 0.0, 0.0
        self._last.pop(key, None)
        return None

    def handle_input(self, window, real_dt):
        # keyboard zoom
        zoom = 0.0
        if window.is_pressed(ti.ui.UP) or window.is_pressed("w") or window.is_pressed("="):
            zoom -= ZOOM_KEY_RATE * real_dt
        if window.is_pressed(ti.ui.DOWN) or window.is_pressed("s") or window.is_pressed("-"):
            zoom += ZOOM_KEY_RATE * real_dt
        shift = window.is_pressed(ti.ui.SHIFT)
        # left drag: orbit (or zoom with shift)
        d = self._drag(window, ti.ui.LMB, "lmb")
        if d is not None and (d[0] != 0.0 or d[1] != 0.0):
            if shift:
                zoom -= d[1] * ZOOM_DRAG_SENS
            else:
                self.target[0] -= d[0] * ORBIT_SENS
                self.target[1] = min(MAX_PITCH, max(-MAX_PITCH, self.target[1] + d[1] * PITCH_SENS))
                self.name = "FREE ORBIT"
        # middle drag: zoom
        d = self._drag(window, ti.ui.MMB, "mmb")
        if d is not None and d[1] != 0.0:
            zoom -= d[1] * ZOOM_DRAG_SENS
        # right drag: pan in the view plane
        d = self._drag(window, ti.ui.RMB, "rmb")
        if d is not None and (d[0] != 0.0 or d[1] != 0.0):
            right, up = self._view_axes()
            span = 2.0 * self.dist * math.tan(math.radians(45.0) / 2.0)     # visible height at the focus
            self.pan_target -= (d[0] * right + d[1] * up) * span * PAN_SENS
            self.name = "FREE ORBIT"
        if zoom != 0.0:
            self.target[2] = min(MAX_DIST, max(MIN_DIST, self.target[2] * math.exp(zoom)))

    # ------------------------------------------------------------ geometry
    def _view_axes(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        forward = -np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), sp])
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
        right = right / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        up = np.cross(right, forward)
        return right, up

    def ease(self, real_dt):
        """Move smoothly toward the target view (visual only)."""
        k = 1.0 - math.exp(-EASE_RATE * real_dt)
        self.yaw += (self.target[0] - self.yaw) * k
        self.pitch += (self.target[1] - self.pitch) * k
        self.dist += (self.target[2] - self.dist) * k
        self.pan += (self.pan_target - self.pan) * k
        self.focus += (self.pan - self.focus) * k

    def apply(self, scene):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        offset = np.array([self.dist * cp * math.cos(self.yaw), self.dist * cp * math.sin(self.yaw), self.dist * sp])
        self.position = self.focus + offset
        self.cam.position(*self.position)
        self.cam.lookat(*self.focus)
        self.cam.up(0.0, 0.0, 1.0)
        self.cam.fov(45)
        scene.set_camera(self.cam)
        return tuple(self.position)

    def state(self):
        return {"name": self.name, "yaw": self.yaw, "pitch": self.pitch, "dist": self.dist,
                "focus": self.focus.copy(), "position": self.position.copy()}
