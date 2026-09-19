"""Unit tests for the state-transition and activity models."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                                 # noqa: E402
from physics import states                                           # noqa: E402
from physics.activity import Activity, log_index, rotation_intensity, accretion_intensity   # noqa: E402
from physics.model import SystemModel                                # noqa: E402

MD = 1e10   # kg/s, above ACCRETION_FLOOR


class TestClassify(unittest.TestCase):
    def test_bands_from_config(self):
        lo, hi = cfg.STATE_ACCRETING_MAX_FASTNESS, cfg.STATE_PROPELLER_MIN_FASTNESS
        self.assertEqual(states.classify(MD, 1e4, 8e4, lo - 1e-6), states.ACCRETING)
        self.assertEqual(states.classify(MD, 1e4, 8e4, lo), states.TRANSITION)
        self.assertEqual(states.classify(MD, 1e4, 8e4, hi), states.TRANSITION)
        self.assertEqual(states.classify(MD, 1e4, 8e4, hi + 1e-6), states.LOW_ACCRETION)

    def test_pulsar_rules(self):
        self.assertEqual(states.classify(0.0, 1e4, 8e4, 0.1), states.PULSAR_DOMINATED)          # no feeding
        self.assertEqual(states.classify(MD, 8e4, 8e4, 0.1), states.PULSAR_DOMINATED)           # ejector r_m >= r_lc
        self.assertEqual(states.classify(MD, math.inf, 8e4, math.inf), states.PULSAR_DOMINATED)

    def test_all_states_have_colours_and_ranks(self):
        for s in states.ALL_STATES:
            self.assertIn(s, states.STATE_COLORS)
            self.assertIn(s, states.STATE_RANK)
        self.assertIs(states.PULSAR, states.PULSAR_DOMINATED)
        self.assertIs(states.LOW, states.LOW_ACCRETION)


class TestStateMachine(unittest.TestCase):
    def test_history_and_direction(self):
        sm = states.StateMachine()
        self.assertEqual(sm.state, states.PULSAR_DOMINATED)
        self.assertEqual(sm.update(10.0, MD, 1e4, 8e4, 0.5), states.ACCRETING)
        self.assertEqual(sm.n_transitions, 1)
        self.assertEqual(sm.direction(), +1)
        self.assertEqual(sm.time_in_state(25.0), 15.0)
        sm.update(30.0, MD, 1e4, 8e4, 0.5)                      # unchanged -> no transition
        self.assertEqual(sm.n_transitions, 1)
        sm.update(40.0, MD, 1e4, 8e4, 2.0)
        self.assertEqual(sm.state, states.LOW_ACCRETION)
        self.assertEqual(sm.direction(), -1)
        self.assertEqual(sm.transitions[-1], (40.0, states.ACCRETING, states.LOW_ACCRETION))

    def test_history_bounded(self):
        sm = states.StateMachine()
        for k in range(200):
            sm.update(float(k), MD, 1e4, 8e4, 0.5 if k % 2 else 2.0)
        self.assertLessEqual(len(sm.transitions), cfg.STATE_HISTORY_LENGTH)
        self.assertEqual(sm.n_transitions, 200)

    def test_reset(self):
        sm = states.StateMachine()
        sm.update(1.0, MD, 1e4, 8e4, 0.5)
        sm.reset()
        self.assertEqual((sm.state, sm.n_transitions, sm.transitions), (states.PULSAR_DOMINATED, 0, []))


class TestModelStates(unittest.TestCase):
    def test_full_cycle_from_physics(self):
        m = SystemModel()
        m.control = 1.0
        for _ in range(120):
            m.advance(cfg.HOUR)
        self.assertEqual(m.state, states.ACCRETING)
        m.control = 0.0
        for _ in range(300):
            m.advance(cfg.HOUR)
        self.assertEqual(m.state, states.PULSAR_DOMINATED)
        seq = [a for _, a, _ in m.state_machine.transitions] + [m.state]
        self.assertEqual(seq, [states.PULSAR_DOMINATED, states.LOW_ACCRETION, states.TRANSITION, states.ACCRETING,
                               states.TRANSITION, states.LOW_ACCRETION, states.PULSAR_DOMINATED])

    def test_state_not_a_ui_switch(self):
        m = SystemModel()
        m.control = 1.0
        s0 = m.state
        m._refresh_derived()                                    # control changed, physics not advanced
        self.assertEqual(m.state, s0)
        self.assertEqual(m.feeding, "HEAVY")                     # the label follows the control, the state does not

    def test_reset_clears_history(self):
        m = SystemModel()
        m.control = 1.0
        for _ in range(48):
            m.advance(cfg.HOUR)
        self.assertGreater(m.state_machine.n_transitions, 0)
        m.reset()
        self.assertEqual(m.state_machine.n_transitions, 0)


class TestActivity(unittest.TestCase):
    def test_log_index(self):
        self.assertEqual(log_index(0.0), 0.0)
        self.assertEqual(log_index(float("nan")), 0.0)
        self.assertEqual(log_index(cfg.ACTIVITY_L_LOW), 0.0)
        self.assertEqual(log_index(cfg.ACTIVITY_L_HIGH), 1.0)
        self.assertAlmostEqual(log_index(math.sqrt(cfg.ACTIVITY_L_LOW * cfg.ACTIVITY_L_HIGH)), 0.5)

    def test_channel_intensities(self):
        self.assertAlmostEqual(rotation_intensity(cfg.SPINDOWN_LUMINOSITY_OBS, states.PULSAR_DOMINATED), 1.0)
        self.assertAlmostEqual(rotation_intensity(cfg.SPINDOWN_LUMINOSITY_OBS, states.ACCRETING), cfg.ACTIVITY_QUENCH_WITH_DISK)
        L_max = cfg.G * cfg.NS_MASS_OBS * cfg.MAX_TRANSFER_RATE / cfg.NS_RADIUS
        self.assertAlmostEqual(accretion_intensity(0.25 * L_max, cfg.NS_MASS_OBS, cfg.NS_RADIUS), 0.5)
        self.assertEqual(accretion_intensity(-1.0, cfg.NS_MASS_OBS, cfg.NS_RADIUS), 0.0)

    def test_activity_drives_consumers(self):
        fed = SystemModel(); fed.control = 1.0
        for _ in range(300):
            fed.advance(cfg.HOUR)
        starved = SystemModel(); starved.control = 0.0
        starved.advance(cfg.DAY)
        self.assertGreater(fed.activity, starved.activity)
        self.assertEqual(fed.beam_orange, fed.activity_model.accretion_intensity)
        self.assertEqual(fed.beam_blue, fed.activity_model.rotation_intensity)
        self.assertGreater(fed.outflow.polar_index, starved.outflow.polar_index)
        self.assertGreater(fed.activity_model.disk, starved.activity_model.disk)
        t = fed.telemetry()
        self.assertAlmostEqual(t["activity"], log_index(t["L_total_W"]))
        self.assertIn("activity_disk", t)


if __name__ == "__main__":
    unittest.main()
