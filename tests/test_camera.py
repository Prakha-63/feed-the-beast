"""Camera is visualization-only: independent of physics and unable to affect it."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCameraIndependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import taichi as ti
        ti.init(arch=ti.cpu)
        from render.camera import OrbitCamera, PRESETS
        cls.OrbitCamera, cls.PRESETS = OrbitCamera, PRESETS

    def test_module_has_no_physics_imports(self):
        import render.camera as cam
        src = open(cam.__file__, encoding="utf-8").read()
        body = src.split('"""', 2)[2]
        self.assertNotIn("physics", body)
        self.assertNotIn("model", body)

    def test_presets(self):
        c = self.OrbitCamera()
        names = {"1": "FREE ORBIT", "2": "NS CLOSE-UP", "3": "DISK VIEW", "4": "TOP VIEW", "5": "SIDE VIEW", "6": "FAR VIEW"}
        for k, n in names.items():
            c.set_preset(k)
            self.assertEqual(c.name, n)
            np.testing.assert_allclose(c.target, self.PRESETS[k][1:4])
        c.set_preset("x")
        self.assertEqual(c.name, "FAR VIEW")                           # unknown key ignored

    def test_ease_converges(self):
        c = self.OrbitCamera()
        c.set_preset("3")
        for _ in range(300):
            c.ease(1 / 30)
        self.assertAlmostEqual(c.dist, 55.0, places=6)
        np.testing.assert_allclose(c.focus, [0.0, 0.0, 0.0], atol=1e-6)

    def test_pan_and_recentre(self):
        c = self.OrbitCamera()
        c.pan_target[:] = [1.0, 2.0, 0.0]
        for _ in range(300):
            c.ease(1 / 30)
        np.testing.assert_allclose(c.focus, [1.0, 2.0, 0.0], atol=1e-6)
        c.recentre()
        for _ in range(300):
            c.ease(1 / 30)
        np.testing.assert_allclose(c.focus, [0.0, 0.0, 0.0], atol=1e-6)

    def test_view_axes_orthonormal(self):
        c = self.OrbitCamera()
        for yaw, pitch in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.5), (3.0, -1.0)):
            c.yaw, c.pitch = yaw, pitch
            r, u = c._view_axes()
            self.assertAlmostEqual(np.linalg.norm(r), 1.0, places=9)
            self.assertAlmostEqual(np.linalg.norm(u), 1.0, places=9)
            self.assertAlmostEqual(np.dot(r, u), 0.0, places=9)

    def test_camera_cannot_change_physics(self):
        """Identical physics with and without heavy camera manipulation."""
        from physics.model import SystemModel
        import config as cfg
        a, b = SystemModel(), SystemModel()
        a.control = b.control = 0.7
        cam = self.OrbitCamera()
        for i in range(200):
            a.advance(cfg.HOUR)
            b.advance(cfg.HOUR)
            cam.set_preset("123456"[i % 6])
            cam.pan_target[:] = np.sin(i) * 5.0
            cam.target[2] = 3.0 + (i % 50)
            cam.ease(1 / 30)
        self.assertEqual(a.ns.J, b.ns.J)
        self.assertEqual(a.ns.mass, b.ns.mass)
        self.assertEqual(a.disk.mass, b.disk.mass)
        self.assertEqual(a.state, b.state)


if __name__ == "__main__":
    unittest.main()
