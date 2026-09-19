"""Unit tests for the ring accretion disk (physics/disk.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                     # noqa: E402
from physics.disk import AccretionDisk, N_RINGS          # noqa: E402
from physics.model import SystemModel                    # noqa: E402

M = cfg.NS_MASS_OBS
RM_FIXED = lambda mdot: 3.0e4     # noqa: E731 - constant truncation for bookkeeping tests


class TestRings(unittest.TestCase):
    def setUp(self):
        self.d = AccretionDisk(cfg.NS_RADIUS, 3.0e8)

    def test_grid_and_keplerian_rotation(self):
        d = self.d
        self.assertEqual(len(d.r), N_RINGS)
        self.assertAlmostEqual(d.edges[0], cfg.NS_RADIUS)
        self.assertAlmostEqual(d.edges[-1], 3.0e8)
        np.testing.assert_allclose(d.omega(M), np.sqrt(cfg.G * M / d.r ** 3), rtol=1e-14)
        self.assertTrue(np.all(np.diff(d.omega(M)) < 0))

    def test_viscous_time_scaling(self):
        d = self.d
        self.assertAlmostEqual(d.t_visc_ring.sum(), cfg.DISK_VISCOUS_TIME)
        ratio = d.t_visc_ring[1:] / d.t_visc_ring[:-1]
        np.testing.assert_allclose(ratio, (d.r[1:] / d.r[:-1]) ** 1.5, rtol=1e-12)

    def test_mass_conservation_any_dt(self):
        for dt in (1.0, 600.0, cfg.DAY, 100 * cfg.YEAR):
            d = AccretionDisk(cfg.NS_RADIUS, 3.0e8)
            stored_expected = 0.0
            for i in range(20):
                mdot = cfg.MAX_TRANSFER_RATE if i < 10 else 0.0
                out = d.step(mdot, dt, M, RM_FIXED)
                stored_expected += (mdot - out) * dt
            self.assertAlmostEqual(d.mass, stored_expected, delta=1e-9 * cfg.MAX_TRANSFER_RATE * 10 * dt)
            self.assertTrue(np.all(d.m >= 0))

    def test_steady_state_throughput(self):
        d = self.d
        for _ in range(400):
            d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
        self.assertAlmostEqual(d.mdot_mag / cfg.MAX_TRANSFER_RATE, 1.0, places=6)
        self.assertTrue(np.all(d.m[d.active()] > 0))
        self.assertTrue(np.all(d.m[~d.active()] == 0))

    def test_persistence_and_gradual_drain(self):
        d = self.d
        for _ in range(400):
            d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
        m0 = d.mass
        masses = []
        for _ in range(72):
            d.step(0.0, cfg.HOUR, M, RM_FIXED)
            masses.append(d.mass)
        self.assertGreater(masses[0], 0.9 * m0)                       # nothing vanishes instantly
        self.assertTrue(all(b < a for a, b in zip(masses, masses[1:])))
        self.assertLess(masses[-1], 0.05 * m0)
        self.assertGreater(masses[-1], 0.0)

    def test_truncation_follows_flux(self):
        """A larger r_m sweeps inner rings; the disk edge tracks it."""
        d = self.d
        for _ in range(400):
            d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
        self.assertLess(abs(d.r_in - 3.0e4) / 3.0e4, 0.3)             # grid-resolved
        m_before = d.mass
        d.step(cfg.MAX_TRANSFER_RATE, 1.0, M, lambda mdot: 1.0e6)
        self.assertGreater(d.r_in, 5e5)
        self.assertTrue(np.all(d.m[d.r < d.r_in] == 0))
        self.assertGreater(d.mdot_mag * 1.0, 0.5 * (m_before - d.mass))  # swept mass was handed over

    def test_temperature_profile_and_luminosity(self):
        d = self.d
        for _ in range(400):
            d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
        T = d.temperature(M)
        act = d.active()
        self.assertTrue(np.all(T[~act] == 0))
        self.assertGreater(T[act][1], T[act][-1])
        self.assertAlmostEqual(d.luminosity(M), cfg.G * M * d.mdot_mag / (2 * d.r_in))
        for _ in range(240):                                          # 10 days starved, hourly
            d.step(0.0, cfg.HOUR, M, RM_FIXED)
        self.assertLess(d.luminosity(M), 1e-3 * cfg.G * M * cfg.MAX_TRANSFER_RATE / (2 * d.r_in))

    def test_angular_momentum_budget(self):
        d = self.d
        for _ in range(50):
            J0 = d.angular_momentum(M)
            d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
            dJdt = (d.angular_momentum(M) - J0) / cfg.HOUR
            closure = d.jdot_in - d.jdot_mag - d.jdot_tidal - dJdt      # budget must close every step
            self.assertLess(abs(closure), 1e-9 * d.jdot_in)
        self.assertAlmostEqual(d.jdot_mag, d.mdot_mag * math.sqrt(cfg.G * M * d.r_in))
        self.assertAlmostEqual(d.angular_momentum(M), float(np.sum(d.m * np.sqrt(cfg.G * M * d.r))))

    def test_invalid_dt_and_reset(self):
        d = self.d
        d.step(cfg.MAX_TRANSFER_RATE, cfg.HOUR, M, RM_FIXED)
        m = d.mass
        for dt in (0.0, -1.0, float("nan")):
            d.step(cfg.MAX_TRANSFER_RATE, dt, M, RM_FIXED)
        self.assertEqual(d.mass, m)
        d.reset()
        self.assertEqual(d.mass, 0.0)
        self.assertEqual(d.r_in, d.r_out)


class TestDiskInModel(unittest.TestCase):
    def test_disk_feeds_star_and_torque(self):
        m = SystemModel()
        m.control = 1.0
        for _ in range(400):
            m.advance(cfg.HOUR)
        self.assertAlmostEqual(m.mdot_disk, m.disk.mdot_mag)
        self.assertAlmostEqual(m.disk.r_in / m.tq["r_m"], 1.0, places=9)
        self.assertGreater(m.tq["jdot_acc"], 0.0)
        # the accretion torque uses the disk's specific angular momentum at r_m
        self.assertAlmostEqual(m.tq["jdot_acc"] / (m.disk.jdot_mag * (1 - m.tq["fastness"])), 1.0, places=9)

    def test_no_coupling_oscillation(self):
        """Inner edge must move smoothly while the disk fills (no r_m flip-flop)."""
        m = SystemModel()
        m.control = 0.9
        r_in = []
        for _ in range(300):
            m.advance(120.0)
            r_in.append(m.disk.r_in)
        jumps = np.abs(np.diff(np.log(r_in)))
        self.assertLess(jumps.max(), math.log(3.0))     # never more than one grid decade-ish per step
        self.assertLess(r_in[-1], r_in[0])

    def test_visual_drivers_exposed(self):
        m = SystemModel()
        m.control = 0.7
        m.advance(cfg.HOUR)
        self.assertEqual(len(m.disk_ring_fill), N_RINGS)
        self.assertEqual(len(m.disk_ring_T), N_RINGS)
        self.assertTrue(np.all((m.disk_ring_fill >= 0) & (m.disk_ring_fill <= 1)))
        t = m.telemetry()
        for k in ("disk_J", "disk_r_in_km", "disk_T_max_K", "disk_L_W", "disk_omega_in_rad_s", "disk_jdot_tidal"):
            self.assertIn(k, t)


if __name__ == "__main__":
    unittest.main()
