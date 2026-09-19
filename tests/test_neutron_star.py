"""
Unit tests for the core neutron-star equations.

    python -m unittest discover -s tests -v

Tolerances are relative and stated in each test.  Everything runs in pure
Python (no Taichi, no window).
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                             # noqa: E402
from physics.neutron_star import (NeutronStar, moment_of_inertia, alfven_radius,   # noqa: E402
                                  magnetospheric_radius, corotation_radius,
                                  accretion_torque, loss_torque, spindown_luminosity,
                                  accretion_fraction)
from physics.model import SystemModel                            # noqa: E402

MDOT_HI = 1e-9 * cfg.M_SUN / cfg.YEAR
MDOT_LO = 1e-11 * cfg.M_SUN / cfg.YEAR


class TestDerivedQuantities(unittest.TestCase):
    """Items 6-10: I, Omega, P, f, E_rot as pure consequences of (M, R, J)."""

    def setUp(self):
        self.ns = NeutronStar()
        self.ns.J *= 0.8731            # arbitrary spin so identities are non-trivial

    def test_moment_of_inertia(self):
        I = moment_of_inertia(self.ns.mass, self.ns.radius)
        self.assertAlmostEqual(I / (cfg.MOMENT_OF_INERTIA_PREFACTOR * self.ns.mass * self.ns.radius ** 2), 1.0, places=14)
        self.assertGreater(I, 1e37)      # ~1e38 kg m^2 for a 1.7 Msun, 12 km star
        self.assertLess(I, 1e39)

    def test_omega_equals_J_over_I(self):
        self.assertAlmostEqual(self.ns.omega, self.ns.J / self.ns.inertia, delta=1e-12 * self.ns.omega)

    def test_period(self):
        self.assertAlmostEqual(self.ns.period, 2 * math.pi / self.ns.omega, delta=1e-12 * self.ns.period)

    def test_frequency(self):
        self.assertAlmostEqual(self.ns.frequency * self.ns.period, 1.0, places=13)

    def test_rotational_energy(self):
        E = 0.5 * self.ns.inertia * self.ns.omega ** 2
        self.assertAlmostEqual(self.ns.rotational_energy / E, 1.0, places=13)

    def test_initial_state_matches_observed_period(self):
        ns = NeutronStar()
        self.assertAlmostEqual(ns.period, cfg.NS_SPIN_PERIOD_OBS, delta=1e-15)
        self.assertAlmostEqual(ns.mass, cfg.NS_MASS_OBS)

    def test_spin_derived_not_stored(self):
        """Changing J alone must change Omega, P and f (no independent spin state)."""
        f0, P0 = self.ns.frequency, self.ns.period
        self.ns.J *= 2.0
        self.assertAlmostEqual(self.ns.frequency / f0, 2.0, places=12)
        self.assertAlmostEqual(self.ns.period / P0, 0.5, places=12)


class TestTorqueModel(unittest.TestCase):
    """Items 2, 4, 5: Mdot -> Jdot_acc, Jdot_loss."""

    def test_accretion_torque_formula(self):
        ns = NeutronStar()
        tq = ns.torques(MDOT_HI)
        expected = MDOT_HI * math.sqrt(cfg.G * ns.mass * tq["r_m"]) * (1.0 - tq["fastness"])
        self.assertAlmostEqual(tq["jdot_acc"] / expected, 1.0, places=12)

    def test_torque_sign_around_corotation(self):
        ns = NeutronStar()
        mdot = 0.5 * MDOT_HI
        r_m = ns.torques(mdot)["r_m"]
        om_eq = math.sqrt(cfg.G * ns.mass / r_m ** 3)          # r_co == r_m
        slow = NeutronStar(J=ns.inertia * om_eq * 0.5)
        fast = NeutronStar(J=ns.inertia * om_eq * 1.5)
        eq = NeutronStar(J=ns.inertia * om_eq)
        self.assertGreater(slow.torques(mdot)["jdot_acc"], 0.0)          # spin-up
        self.assertLess(fast.torques(mdot)["jdot_acc"], 0.0)             # propeller braking
        self.assertAlmostEqual(eq.torques(mdot)["fastness"], 1.0, places=9)
        self.assertAlmostEqual(eq.torques(mdot)["jdot_acc"], 0.0, delta=1e-6 * abs(slow.torques(mdot)["jdot_acc"]))

    def test_more_feeding_more_spinup(self):
        ns = NeutronStar()
        lo, hi = ns.torques(0.3 * MDOT_HI), ns.torques(MDOT_HI)
        self.assertGreater(hi["jdot_acc"], lo["jdot_acc"])
        self.assertGreater(hi["mdot_ns"], lo["mdot_ns"])
        self.assertLess(hi["r_m"], lo["r_m"])                           # r_m ~ Mdot^-2/7

    def test_alfven_scaling(self):
        ns = NeutronStar()
        r1 = alfven_radius(ns.B, ns.radius, ns.mass, MDOT_LO)
        r2 = alfven_radius(ns.B, ns.radius, ns.mass, 128 * MDOT_LO)
        self.assertAlmostEqual(r1 / r2, 128 ** (2 / 7), places=9)

    def test_magnetosphere_never_inside_star(self):
        ns = NeutronStar()
        self.assertGreaterEqual(magnetospheric_radius(ns.B, ns.radius, ns.mass, 1e6 * MDOT_HI), ns.radius)

    def test_loss_torque_is_dipole_braking(self):
        ns = NeutronStar()
        L = spindown_luminosity(ns.B, ns.radius, ns.omega, ns.alpha)
        self.assertAlmostEqual(loss_torque(ns.B, ns.radius, ns.omega, ns.alpha) * ns.omega / L, 1.0, places=12)
        self.assertGreater(L, 0.0)
        # L_sd ~ Omega^4
        self.assertAlmostEqual(spindown_luminosity(ns.B, ns.radius, 2 * ns.omega, ns.alpha) / L, 16.0, places=9)

    def test_loss_matches_observed_order_of_magnitude(self):
        L = NeutronStar().torques(0.0)["L_sd"]
        self.assertLess(abs(math.log10(L / cfg.SPINDOWN_LUMINOSITY_OBS)), math.log10(3))

    def test_accretion_fraction_switch(self):
        self.assertGreater(accretion_fraction(0.3), 0.99)
        self.assertLess(accretion_fraction(2.0), 0.01)
        self.assertAlmostEqual(accretion_fraction(1.0), 0.5, places=12)


class TestEvolution(unittest.TestCase):
    """Items 1-3: dM/dt = f_acc*Mdot, dJ/dt = Jdot_acc - Jdot_loss; accretion drives spin."""

    @staticmethod
    def steady(control, hours=400):
        m = SystemModel()
        m.control = control
        for _ in range(hours):
            m.advance(cfg.HOUR)
        return m

    def test_mass_evolution(self):
        m = self.steady(1.0)
        M0, T = m.ns.mass, 1000 * cfg.YEAR
        m.advance(T)
        self.assertAlmostEqual((m.ns.mass - M0) / (m.tq["f_acc"] * m.mdot_in * T), 1.0, places=6)

    def test_angular_momentum_increment(self):
        ns = NeutronStar()
        tq = ns.torques(0.5 * MDOT_HI)
        J0, dt = ns.J, cfg.YEAR
        ns.apply(tq["mdot_ns"] * dt, (tq["jdot_acc"] - tq["jdot_loss"]) * dt)
        expected = (tq["jdot_acc"] - tq["jdot_loss"]) * dt
        # J ~ 7e41 and dJ ~ 3e33: compare at double-precision granularity of J
        self.assertAlmostEqual(ns.J - J0, expected, delta=1e-15 * J0)

    def test_feeding_spins_up_starving_spins_down(self):
        fed = self.steady(1.0)
        starved = self.steady(0.0)
        f_fed0, f_st0 = fed.ns.frequency, starved.ns.frequency
        for m in (fed, starved):
            m.advance(1e4 * cfg.YEAR)
        self.assertGreater(fed.ns.frequency, f_fed0)
        self.assertLess(starved.ns.frequency, f_st0)
        self.assertGreater(fed.ns.mass, cfg.NS_MASS_OBS)
        self.assertEqual(starved.ns.mass, cfg.NS_MASS_OBS)

    def test_zero_accretion_has_no_accretion_torque(self):
        m = self.steady(0.0, hours=24)
        self.assertEqual(m.tq["jdot_acc"], 0.0)
        self.assertEqual(m.tq["mdot_ns"], 0.0)
        self.assertGreater(m.tq["jdot_loss"], 0.0)

    def test_frequency_follows_J_not_time(self):
        """Pausing (dt = 0) leaves f unchanged; only J changes f."""
        m = self.steady(1.0)
        f0 = m.ns.frequency
        m.advance(0.0)
        self.assertEqual(m.ns.frequency, f0)
        m.ns.J *= 1.01
        m._refresh_derived()
        self.assertAlmostEqual(m.ns.frequency / f0, 1.01, places=12)

    def test_timestep_configurable_and_consistent(self):
        """Same physical interval with different step sizes / substeps agrees to 1e-3."""
        T = 100 * cfg.YEAR
        a = self.steady(0.9); a.substeps = 2
        b = self.steady(0.9); b.substeps = 32
        for _ in range(100):
            a.advance(T / 100)
        b.advance(T)
        dJa, dJb = a.ns.J - NeutronStar().J, b.ns.J - NeutronStar().J
        self.assertAlmostEqual(dJa / dJb, 1.0, places=3)

    def test_time_warp_does_not_change_physics(self):
        T = 30 * cfg.DAY
        a, b = SystemModel(), SystemModel()
        a.control = b.control = 0.7
        a.warp_index, b.warp_index = 4, 5           # 1 day/s vs 1 yr/s
        while a.sim_time < T - 1e-6:
            a.step(min(1 / 30, (T - a.sim_time) / a.warp))
        while b.sim_time < T - 1e-6:
            b.step(min(1 / 30, (T - b.sim_time) / b.warp))
        self.assertAlmostEqual(a.sim_time, b.sim_time, delta=1e-6 * T)
        self.assertAlmostEqual(a.ns.J / b.ns.J, 1.0, places=9)
        self.assertAlmostEqual(a.ns.mass / b.ns.mass, 1.0, places=9)


class TestSafeguards(unittest.TestCase):
    def test_invalid_inertia_inputs(self):
        with self.assertRaises(ValueError):
            moment_of_inertia(0.0, 1e4)
        with self.assertRaises(ValueError):
            moment_of_inertia(1e30, -1.0)

    def test_non_rotating_star(self):
        ns = NeutronStar(J=1e-300)
        ns.J = 0.0
        self.assertEqual(ns.omega, 0.0)
        self.assertTrue(math.isinf(ns.period))
        self.assertEqual(ns.frequency, 0.0)
        self.assertEqual(ns.rotational_energy, 0.0)
        tq = ns.torques(MDOT_HI)
        self.assertTrue(math.isinf(tq["r_co"]) and math.isinf(tq["r_lc"]))
        self.assertEqual(tq["jdot_loss"], 0.0)
        self.assertGreater(tq["jdot_acc"], 0.0)          # accretion can spin it up
        self.assertTrue(all(math.isfinite(v) for k, v in tq.items() if k not in ("r_co", "r_lc")))

    def test_zero_and_negative_mdot(self):
        ns = NeutronStar()
        for mdot in (0.0, -1.0, float("nan")):
            tq = ns.torques(mdot)
            self.assertEqual(tq["jdot_acc"], 0.0)
            self.assertEqual(tq["mdot_ns"], 0.0)
            self.assertEqual(tq["L_acc"], 0.0)
            self.assertTrue(math.isinf(tq["r_m"]))

    def test_J_cannot_go_negative(self):
        ns = NeutronStar()
        ns.apply(0.0, -10 * ns.J)
        self.assertEqual(ns.J, 0.0)

    def test_mass_cannot_decrease(self):
        ns = NeutronStar()
        ns.apply(-1e20, 0.0)
        self.assertEqual(ns.mass, cfg.NS_MASS_OBS)

    def test_non_finite_increment_rejected(self):
        with self.assertRaises(ValueError):
            NeutronStar().apply(float("inf"), 0.0)

    def test_model_ignores_invalid_dt(self):
        m = SystemModel()
        m.control = 1.0
        for dt in (0.0, -5.0, float("nan"), float("inf")):
            m.advance(dt)
        self.assertEqual(m.sim_time, 0.0)
        self.assertEqual(m.ns.mass, cfg.NS_MASS_OBS)

    def test_extreme_warp_stays_finite(self):
        m = SystemModel()
        m.control = 1.0
        for _ in range(50):
            m.advance(1e6 * cfg.YEAR)
        t = m.telemetry()
        self.assertTrue(all(math.isfinite(v) for v in (t["mass_msun"], t["freq_hz"], t["period_ms"], t["J"], t["E_rot_J"])))
        self.assertLess(m.ns.omega, m.ns.breakup_omega * 1.5)


if __name__ == "__main__":
    unittest.main()
