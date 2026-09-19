"""Unit tests for the idealized dipole model (physics/magnetosphere.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                                     # noqa: E402
from physics.magnetosphere import (magnetic_axis, magnetic_basis, field_line, field_line_points,  # noqa: E402
                                   field_strength, field_vector, dipole_moment, shell_parameters,
                                   Magnetosphere, SPIN_AXIS)
from physics.neutron_star import NeutronStar                             # noqa: E402
from physics.model import SystemModel                                    # noqa: E402


class TestAxes(unittest.TestCase):
    def test_axis_geometry(self):
        for alpha in (0.0, 0.3, math.radians(35), math.pi / 2):
            for ph in (0.0, 1.0, 4.0):
                m = magnetic_axis(ph, alpha)
                self.assertAlmostEqual(np.linalg.norm(m), 1.0, places=14)
                self.assertAlmostEqual(math.acos(np.clip(np.dot(m, SPIN_AXIS), -1, 1)), alpha, places=12)
                if alpha > 0:
                    self.assertAlmostEqual((math.atan2(m[1], m[0]) - ph) % (2 * math.pi) % (2 * math.pi), 0.0, places=12)

    def test_basis_orthonormal(self):
        for alpha in (0.0, 0.6, math.pi / 2):
            e1, e2, m = magnetic_basis(0.9, alpha)
            M = np.stack([e1, e2, m])
            np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.cross(e1, e2), m, atol=1e-12)     # right-handed

    def test_misalignment_configurable(self):
        ns = NeutronStar(alpha=math.radians(10.0))
        mg = Magnetosphere(ns)
        self.assertAlmostEqual(mg.telemetry(1e5, 1e5)["misalignment_deg"], 10.0)
        self.assertAlmostEqual(math.degrees(math.acos(mg.magnetic_axis[2])), 10.0, places=10)

    def test_axis_rotates_with_rotation_state_only(self):
        m = SystemModel(sim_seconds_per_real_second=1e-3)
        a0 = m.magnetosphere.magnetic_axis
        m.step(1.0)                                                   # 1 ms
        a1 = m.magnetosphere.magnetic_axis
        turned = (math.atan2(a1[1], a1[0]) - math.atan2(a0[1], a0[0])) % (2 * math.pi)
        self.assertAlmostEqual(turned, (m.ns.omega * 1e-3) % (2 * math.pi), places=9)
        m.ns.J *= 2.0                                                 # faster spin -> faster sweep, same dt
        m.ns.spin_phase = 0.0
        m.step(1.0)
        self.assertAlmostEqual(m.ns.spin_phase, (2 * m.ns.omega / 2 * 1e-3) % (2 * math.pi), places=9)
        m.paused = True
        a2 = m.magnetosphere.magnetic_axis
        m.step(1.0)
        self.assertTrue(np.array_equal(a2, m.magnetosphere.magnetic_axis))


class TestDipole(unittest.TestCase):
    def setUp(self):
        self.ns = NeutronStar()

    def test_field_line_shape(self):
        R, L = self.ns.radius, 4 * self.ns.radius
        tr = field_line(L, R, 101)
        th, r = tr[:, 0], tr[:, 1]
        np.testing.assert_allclose(r, L * np.sin(th) ** 2, rtol=1e-12)
        self.assertAlmostEqual(r[0] / R, 1.0, places=9)
        self.assertAlmostEqual(r[-1] / R, 1.0, places=9)
        self.assertAlmostEqual(r.max() / L, 1.0, places=4)
        self.assertEqual(len(field_line(0.5 * R, R, 10)), 0)         # no line inside the star

    def test_field_line_points_follow_axis(self):
        basis = magnetic_basis(0.4, self.ns.alpha)
        pts = field_line_points(3 * self.ns.radius, self.ns.radius, 51, 0.0, basis)
        foot = pts[0]                                                  # northern footpoint: cos(theta0) = sqrt(1 - R/L)
        self.assertAlmostEqual(np.dot(foot, basis[2]) / np.linalg.norm(foot), math.sqrt(1 - 1 / 3), places=9)
        self.assertAlmostEqual(np.linalg.norm(foot) / self.ns.radius, 1.0, places=9)
        eq = pts[np.argmax(np.linalg.norm(pts, axis=1))]
        self.assertAlmostEqual(abs(np.dot(eq, basis[2])) / np.linalg.norm(eq), 0.0, places=2)   # apex in the magnetic equator

    def test_field_strength_and_moment(self):
        B, R = self.ns.B, self.ns.radius
        self.assertAlmostEqual(field_strength(B, R, R, math.pi / 2), B)
        self.assertAlmostEqual(field_strength(B, R, R, 0.0), 2 * B)
        self.assertAlmostEqual(field_strength(B, R, 2 * R, math.pi / 2) / B, 1 / 8, places=14)
        mu = dipole_moment(B, R)
        self.assertAlmostEqual(cfg.MU0 * mu / (4 * math.pi * R ** 3), B)
        basis = magnetic_basis(0.0, self.ns.alpha)
        Bv = field_vector(B, R, 5 * R * basis[0], basis)          # equatorial point
        self.assertAlmostEqual(np.linalg.norm(Bv) / (B / 125), 1.0, places=12)
        self.assertLess(np.dot(Bv, basis[2]), 0)                 # antiparallel to the axis at the equator

    def test_shells(self):
        sh = shell_parameters(self.ns.radius, 50e3, 10)
        self.assertEqual(len(sh), 10)
        self.assertAlmostEqual(sh[-1], 50e3)
        self.assertGreater(sh[0], 1.05 * self.ns.radius)
        sh2 = shell_parameters(self.ns.radius, 1.0, 5)             # r_max inside the star -> still valid
        self.assertTrue(np.all(np.isfinite(sh2)) and np.all(sh2 > self.ns.radius))

    def test_telemetry_exposes_model_field(self):
        m = SystemModel()
        t = m.telemetry()
        self.assertAlmostEqual(t["B_surface_G"], cfg.B_SURFACE_OBS / cfg.GAUSS)
        self.assertAlmostEqual(t["B_polar_G"], 2 * cfg.B_SURFACE_OBS / cfg.GAUSS)
        self.assertEqual(t["misalignment_deg"], math.degrees(cfg.MAGNETIC_MISALIGNMENT))
        np.testing.assert_allclose(t["magnetic_axis"], m.magnetosphere.magnetic_axis)
        m.ns.accreted_mass = cfg.FIELD_BURIAL_MASS                    # burial halves B
        self.assertAlmostEqual(m.telemetry()["B_surface_G"], 0.5 * cfg.B_SURFACE_OBS / cfg.GAUSS)


if __name__ == "__main__":
    unittest.main()
