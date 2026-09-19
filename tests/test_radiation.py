"""Unit tests for the pulsar radiation model (physics/radiation.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                                          # noqa: E402
from physics import states                                                    # noqa: E402
from physics.neutron_star import NeutronStar                                  # noqa: E402
from physics.radiation import PulsarRadiation, BEAM_HALF_ANGLE, beam_pattern  # noqa: E402
from physics.model import SystemModel                                         # noqa: E402


class TestGeometry(unittest.TestCase):
    def test_opposed_beams_on_magnetic_axis(self):
        ns = NeutronStar()
        ns.spin_phase = 1.3
        r = PulsarRadiation(ns)
        d1, d2 = r.directions()
        np.testing.assert_allclose(d1, -d2)
        self.assertAlmostEqual(math.acos(d1[2]), ns.alpha, places=12)
        self.assertAlmostEqual(math.atan2(d1[1], d1[0]), 1.3, places=12)

    def test_beam_follows_rotation_state(self):
        m = SystemModel(sim_seconds_per_real_second=1e-3)
        d0, _ = m.radiation.directions()
        m.step(0.25)
        d1, _ = m.radiation.directions()
        swept = (math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])) % (2 * math.pi)
        self.assertAlmostEqual(swept, (m.ns.omega * 0.25e-3) % (2 * math.pi), places=9)
        m.ns.J *= 0.5                          # slower star -> slower sweep for the same dt
        m.ns.spin_phase = 0.0
        m.step(0.25)
        self.assertAlmostEqual(m.ns.spin_phase, (m.ns.omega * 0.25e-3) % (2 * math.pi), places=9)

    def test_beam_pattern(self):
        self.assertAlmostEqual(beam_pattern(1.0), 1.0)
        self.assertAlmostEqual(beam_pattern(math.cos(BEAM_HALF_ANGLE)), math.exp(-1.0))
        self.assertLess(beam_pattern(math.cos(3 * BEAM_HALF_ANGLE)), 1e-3)


class TestPulses(unittest.TestCase):
    def setUp(self):
        self.ns = NeutronStar()
        self.r = PulsarRadiation(self.ns)
        self.r.update(dict(L_sd=cfg.SPINDOWN_LUMINOSITY_OBS, L_acc=0.0), states.PULSAR)

    def test_period_and_frequency_from_rotation(self):
        self.assertEqual(self.r.pulse_period, self.ns.period)
        self.assertEqual(self.r.pulse_frequency, self.ns.frequency)
        self.ns.J *= 2.0
        self.assertAlmostEqual(self.r.pulse_frequency / (2 / cfg.NS_SPIN_PERIOD_OBS), 1.0, places=12)

    def test_one_pulse_per_rotation_at_default_geometry(self):
        self.assertEqual(self.r.pulses_per_rotation, 1)
        ph = np.linspace(0, 2 * math.pi, 3601, endpoint=False)
        I = self.r.profile(ph)
        self.assertAlmostEqual(ph[int(np.argmax(I))] % (2 * math.pi), 0.0, delta=2 * math.pi / 3600)
        above = I > 0.5
        edges = np.sum(above[1:] != above[:-1])
        self.assertEqual(edges, 2)                                    # a single peak crossing half maximum

    def test_interpulse_for_orthogonal_rotator(self):
        ns = NeutronStar(alpha=math.pi / 2)
        r = PulsarRadiation(ns, inclination=math.pi / 2)
        r.update(dict(L_sd=cfg.SPINDOWN_LUMINOSITY_OBS, L_acc=0.0), states.PULSAR)
        self.assertEqual(r.pulses_per_rotation, 2)
        ph = np.linspace(0, 2 * math.pi, 3600, endpoint=False)
        I = r.profile(ph)
        prev, nxt = np.roll(I, 1), np.roll(I, -1)                     # circular peak search
        peaks = np.where((I > prev) & (I >= nxt) & (I > 0.5))[0]
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual((ph[peaks[1]] - ph[peaks[0]]) / math.pi, 1.0, places=2)

    def test_time_to_next_pulse(self):
        self.ns.spin_phase = 0.0
        self.assertAlmostEqual(self.r.time_to_next_pulse(), 0.0)
        self.ns.spin_phase = math.pi
        self.assertAlmostEqual(self.r.time_to_next_pulse() / self.ns.period, 0.5, places=12)
        z = NeutronStar(J=1.0)
        z.J = 0.0
        self.assertTrue(math.isinf(PulsarRadiation(z).time_to_next_pulse()))

    def test_window_ends_at_current_phase(self):
        self.ns.spin_phase = 2.0
        ph, I = self.r.profile_window(101, 3)
        self.assertAlmostEqual(ph[-1], 2.0)
        self.assertAlmostEqual(ph[0], 2.0 - 6 * math.pi)
        self.assertEqual(len(I), 101)


class TestIntensity(unittest.TestCase):
    def test_channels(self):
        ns = NeutronStar()
        r = PulsarRadiation(ns)
        r.update(dict(L_sd=0.0, L_acc=0.0), states.PULSAR)
        self.assertEqual(r.intensity, 0.0)
        r.update(dict(L_sd=cfg.SPINDOWN_LUMINOSITY_OBS, L_acc=0.0), states.PULSAR)
        self.assertAlmostEqual(r.I_rot, 1.0)
        self.assertTrue(r.radio_on)
        r.update(dict(L_sd=cfg.SPINDOWN_LUMINOSITY_OBS, L_acc=0.0), states.LOW)
        self.assertAlmostEqual(r.I_rot, 0.25)
        self.assertFalse(r.radio_on)
        L_max = cfg.G * ns.mass * cfg.MAX_TRANSFER_RATE / ns.radius
        r.update(dict(L_sd=0.0, L_acc=L_max), states.ACCRETING)
        self.assertAlmostEqual(r.I_acc, 1.0)
        r.update(dict(L_sd=10 * cfg.SPINDOWN_LUMINOSITY_OBS, L_acc=10 * L_max), states.ACCRETING)
        self.assertEqual(r.intensity, 1.0)                            # clipped

    def test_model_intensity_tracks_feeding(self):
        m = SystemModel()
        m.control = 0.0
        m.advance(cfg.DAY)
        self.assertGreater(m.beam_blue, 0.9)
        self.assertEqual(m.beam_orange, 0.0)
        m.control = 1.0
        for _ in range(200):
            m.advance(cfg.HOUR)
        self.assertGreater(m.beam_orange, 0.8)
        self.assertAlmostEqual(m.beam_blue, 0.25 * min(1.0, m.tq["L_sd"] / cfg.SPINDOWN_LUMINOSITY_OBS))
        t = m.telemetry()
        self.assertEqual(t["pulse_period_ms"], m.ns.period * 1e3)


if __name__ == "__main__":
    unittest.main()
