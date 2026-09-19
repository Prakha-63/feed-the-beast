"""Unit tests for the particle-outflow drivers (physics/outflow.py)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                                  # noqa: E402
from physics.neutron_star import NeutronStar                          # noqa: E402
from physics.outflow import Outflow, polar_cap_angle, log_index, F_JET, V_POLAR   # noqa: E402
from physics.model import SystemModel                                 # noqa: E402


class TestDrivers(unittest.TestCase):
    def test_polar_cap_angle(self):
        R = cfg.NS_RADIUS
        self.assertAlmostEqual(polar_cap_angle(R, 4 * R), math.asin(0.5))
        self.assertGreater(polar_cap_angle(R, 4 * R), polar_cap_angle(R, 100 * R))    # slower spin -> smaller cap
        self.assertEqual(polar_cap_angle(R, math.inf), 0.0)
        self.assertEqual(polar_cap_angle(R, R), math.pi / 2)

    def test_log_index(self):
        self.assertEqual(log_index(0.0), 0.0)
        self.assertEqual(log_index(-5.0), 0.0)
        self.assertEqual(log_index(cfg.ACTIVITY_L_LOW / 10), 0.0)
        self.assertAlmostEqual(log_index(math.sqrt(cfg.ACTIVITY_L_LOW * cfg.ACTIVITY_L_HIGH)), 0.5)
        self.assertEqual(log_index(1e40), 1.0)
        self.assertLess(log_index(1e27), log_index(1e28))                              # monotonic

    def test_update_from_torque_dict(self):
        ns = NeutronStar()
        of = Outflow(ns)
        tq = ns.torques(0.0)
        of.update(tq)
        self.assertAlmostEqual(of.polar_power, tq["L_sd"])
        self.assertEqual(of.polar_rotation_fraction, 1.0)
        self.assertEqual(of.eq_index, 0.0)
        self.assertEqual(of.polar_speed, V_POLAR)
        tq2 = ns.torques(cfg.MAX_TRANSFER_RATE)
        of.update(tq2)
        self.assertAlmostEqual(of.polar_power, tq2["L_sd"] + F_JET * tq2["L_acc"])
        self.assertGreater(of.polar_index, log_index(tq["L_sd"]))
        self.assertAlmostEqual(of.eq_speed, math.sqrt(2 * cfg.G * ns.mass / tq2["r_m"]))

    def test_visual_counts_bounded_and_proportional(self):
        ns = NeutronStar()
        of = Outflow(ns)
        of.update(ns.torques(cfg.MAX_TRANSFER_RATE))
        p, e = of.visual_counts(1000, 1000)
        self.assertEqual(p, int(1000 * of.polar_index))
        self.assertEqual(e, int(1000 * of.eq_index))
        self.assertTrue(0 <= p <= 1000 and 0 <= e <= 1000)
        of.update(dict(L_sd=1e60, L_acc=1e60, mdot_ejected=1e60, r_lc=1e5, r_m=1e5))
        self.assertEqual(of.visual_counts(1000, 1000), (1000, 1000))                    # clipped


class TestActivityRelationship(unittest.TestCase):
    @staticmethod
    def steady(c):
        m = SystemModel()
        m.control = c
        for _ in range(300):
            m.advance(cfg.HOUR)
        return m

    def test_more_activity_more_outflow(self):
        runs = [self.steady(c) for c in (0.0, 0.5, 1.0)]
        acts = [m.activity for m in runs]
        idx = [m.outflow.polar_index for m in runs]
        self.assertTrue(acts[0] < acts[1] < acts[2])
        self.assertTrue(idx[0] < idx[1] < idx[2])

    def test_starving_reduces_outflow(self):
        m = self.steady(1.0)
        i_fed = m.outflow.polar_index
        m.control = 0.0
        for _ in range(240):
            m.advance(cfg.HOUR)
        self.assertLess(m.outflow.polar_index, i_fed)
        self.assertGreater(m.outflow.polar_index, 0.0)                                  # spin-down power remains

    def test_propeller_ejection_channel(self):
        m = self.steady(0.2)
        self.assertGreater(m.outflow.eq_index, 0.0)
        self.assertAlmostEqual(m.outflow.eq_mdot, m.tq["mdot_ejected"])
        fed = self.steady(1.0)
        self.assertLess(fed.outflow.eq_index, m.outflow.eq_index)                      # accretor ejects little

    def test_telemetry(self):
        m = self.steady(0.5)
        t = m.telemetry()
        self.assertAlmostEqual(t["outflow_cap_angle_deg"], math.degrees(m.outflow.cap_angle))
        self.assertAlmostEqual(t["outflow_polar_speed_c"], 0.3)


if __name__ == "__main__":
    unittest.main()
