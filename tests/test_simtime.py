"""Unit tests for the simulation-time system (physics/simtime.py)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                     # noqa: E402
from physics.simtime import SimulationClock, format_duration   # noqa: E402
from physics.model import SystemModel                    # noqa: E402


class TestClock(unittest.TestCase):
    def test_rate_conversion(self):
        c = SimulationClock(3600.0)
        self.assertEqual(c.tick(0.5), 1800.0)
        self.assertEqual(c.sim_time, 1800.0)
        self.assertEqual(c.real_elapsed, 0.5)

    def test_default_rate_from_presets(self):
        c = SimulationClock()
        self.assertEqual(c.rate, cfg.TIME_WARPS[cfg.DEFAULT_WARP_INDEX][0])
        c.step_preset(+1)
        self.assertEqual(c.rate, cfg.TIME_WARPS[cfg.DEFAULT_WARP_INDEX + 1][0])
        c.step_preset(-100)
        self.assertEqual(c.warp_index, 0)

    def test_invalid_rate_rejected(self):
        c = SimulationClock(1.0)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.rate = bad

    def test_invalid_real_dt_ignored(self):
        c = SimulationClock(10.0)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            self.assertEqual(c.tick(bad), 0.0)
        self.assertEqual(c.sim_time, 0.0)

    def test_pause_separates_clocks(self):
        c = SimulationClock(100.0)
        c.tick(1.0)
        c.paused = True
        c.tick(2.0)
        self.assertEqual(c.sim_time, 100.0)
        self.assertEqual(c.real_elapsed, 3.0)
        self.assertEqual(c.real_running, 1.0)

    def test_kinematic_slice(self):
        c = SimulationClock(1e9)
        c.tick(1.0)
        self.assertEqual(c.kin_dt, cfg.KINEMATIC_DT_CAP)
        self.assertTrue(c.kin_capped)
        c = SimulationClock(1.0)
        c.tick(1.0)
        self.assertEqual(c.kin_dt, 1.0)
        self.assertFalse(c.kin_capped)

    def test_breakdown_and_format(self):
        c = SimulationClock(1.0)
        c.advance(3 * cfg.DAY + 4 * cfg.HOUR + 12 * 60 + 7.25)
        b = c.breakdown()
        self.assertEqual((b["days"], b["hours"], b["minutes"]), (3, 4, 12))
        self.assertAlmostEqual(b["seconds"], 7.25)
        self.assertEqual(c.formatted(), "+ 3 d 04:12:07.250")
        self.assertEqual(format_duration(0.004), "+ 4.000 ms")
        self.assertEqual(format_duration(2.5 * cfg.YEAR), "+ 2.500 yr")
        self.assertEqual(format_duration(1.5e6 * cfg.YEAR), "+ 1.500 Myr")

    def test_reset(self):
        c = SimulationClock(5.0)
        c.tick(2.0)
        c.reset()
        self.assertEqual((c.sim_time, c.real_elapsed, c.ticks), (0.0, 0.0, 0))
        self.assertEqual(c.rate, 5.0)


class TestPhysicsUsesSimulationTime(unittest.TestCase):
    def test_fps_independence(self):
        """Different frame rates, same rate and real duration -> same physics."""
        runs = []
        for fps in (24, 60, 144):
            m = SystemModel(sim_seconds_per_real_second=1e3 * cfg.YEAR)
            m.control = 0.9
            for _ in range(200):
                m.advance(cfg.HOUR)                      # steady disk
            for _ in range(fps * 3):
                m.step(1.0 / fps)
            runs.append((m.sim_time, m.ns.mass, m.ns.J))
        for t, M, J in runs[1:]:
            self.assertAlmostEqual(t / runs[0][0], 1.0, places=9)
            self.assertAlmostEqual((M - cfg.NS_MASS_OBS) / (runs[0][1] - cfg.NS_MASS_OBS), 1.0, places=6)
            self.assertAlmostEqual(J / runs[0][2], 1.0, places=12)

    def test_rate_change_only_changes_speed(self):
        """Reaching the same simulated time at two rates gives the same state."""
        out = []
        for rate in (cfg.DAY, 100 * cfg.DAY):
            m = SystemModel(sim_seconds_per_real_second=rate)
            m.control = 0.7
            for _ in range(200):
                m.advance(cfg.HOUR)
            target = m.sim_time + 100 * cfg.DAY
            while m.sim_time < target - 1e-6:
                m.step(min(1 / 30, (target - m.sim_time) / rate))
            out.append((m.sim_time, m.ns.J, m.ns.mass))
        self.assertAlmostEqual(out[0][0], out[1][0], delta=1e-6)
        self.assertAlmostEqual(out[0][1] / out[1][1], 1.0, places=11)
        self.assertAlmostEqual(out[0][2] / out[1][2], 1.0, places=11)

    def test_model_exposes_time_readouts(self):
        m = SystemModel(sim_seconds_per_real_second=3600.0)
        m.step(1.0)
        t = m.telemetry()
        self.assertEqual(t["sim_time_s"], 3600.0)
        self.assertEqual(t["sim_time_formatted"], "+ 01:00:00.000")
        self.assertEqual(t["sim_time_breakdown"]["hours"], 1)
        self.assertEqual(t["sim_seconds_per_real_second"], 3600.0)
        self.assertEqual(t["real_elapsed_s"], 1.0)

    def test_reset_keeps_rate_and_zeroes_time(self):
        m = SystemModel(sim_seconds_per_real_second=42.0)
        m.step(1.0)
        m.reset()
        self.assertEqual(m.sim_time, 0.0)
        self.assertEqual(m.warp, 42.0)


if __name__ == "__main__":
    unittest.main()
