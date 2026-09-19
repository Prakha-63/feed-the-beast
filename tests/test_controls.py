"""
RESET / PAUSE / RESUME / RESTART and deterministic seeding, tested on the
real Simulation object (headless window, CPU backend).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                  # noqa: E402


def visual_snapshot(sim):
    r = sim.renderer
    return {
        "disk_r": r.disk.r.to_numpy(), "disk_phi": r.disk.phi.to_numpy(), "stars": r.stars.pos.to_numpy(),
        "wind_r": r.wind.r.to_numpy(), "wind_dir": r.wind.dirv.to_numpy(), "wind_alive": r.wind.alive.to_numpy(),
        "beam_t": r.beams.t.to_numpy(), "composite_pos": r.composite.pos.to_numpy(),
    }


def physics_snapshot(m):
    return (m.ns.mass, m.ns.J, m.ns.spin_phase, m.disk.mass, m.sim_time, m.state)


class TestControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from simulation import RunConfig, Simulation
        cls.RunConfig, cls.Simulation = RunConfig, Simulation
        cls.sim = Simulation(RunConfig(seed=7, force_cpu=True, show_window=False, control=0.6,
                                       sim_seconds_per_real_second=3600.0, resolution=(320, 180)))

    @classmethod
    def tearDownClass(cls):
        cls.sim.window.destroy()

    def _frames(self, n, dt=1 / 30):
        for _ in range(n):
            self.sim.step_frame(dt)

    # ------------------------------------------------------------ PAUSE / RESUME
    def test_pause_freezes_physics_resume_continues(self):
        s = self.sim
        s.restart()
        self._frames(5)
        s.pause()
        before = physics_snapshot(s.model)
        self._frames(10)
        self.assertEqual(physics_snapshot(s.model), before)          # nothing advanced
        self.assertEqual(s.model.kin_dt, 0.0)                         # visual phases frozen too
        self.assertTrue(s.model.paused)
        s.resume()
        self._frames(1)
        self.assertGreater(s.model.sim_time, before[4])
        self.assertFalse(s.model.paused)
        s.toggle_pause()
        self.assertTrue(s.model.paused)
        s.toggle_pause()
        self.assertFalse(s.model.paused)

    def test_camera_works_while_paused(self):
        s = self.sim
        s.restart()
        s.pause()
        cam = s.renderer.camera
        cam.set_preset("3")
        d0 = cam.dist
        t0 = s.model.sim_time
        self._frames(30)
        self.assertNotEqual(cam.dist, d0)                             # camera eased toward the preset
        self.assertAlmostEqual(cam.dist, 55.0, delta=1.0)
        self.assertEqual(s.model.sim_time, t0)                        # physics still frozen
        s.resume()

    # ------------------------------------------------------------ RESET
    def test_reset_restores_documented_initial_state_keeping_settings(self):
        s = self.sim
        s.restart()
        s.model.set_control(0.9)
        s.model.warp = 86400.0
        self._frames(40)
        self.assertFalse(s.model.is_initial())
        s.reset()
        self.assertTrue(s.model.is_initial())
        init = s.model.initial_state()
        self.assertEqual(s.model.ns.mass, init["mass"])
        self.assertEqual(s.model.ns.J, init["J"])
        self.assertEqual(s.model.disk.mass, 0.0)
        self.assertEqual(s.model.sim_time, 0.0)
        self.assertEqual(s.model.state_machine.n_transitions, 0)
        self.assertEqual(s.model.control, 0.9)                        # user settings kept
        self.assertEqual(s.model.warp, 86400.0)
        self.assertEqual(s.frame, 0)

    def test_reset_while_paused_stays_paused(self):
        s = self.sim
        s.restart()
        self._frames(3)
        s.pause()
        s.reset()
        self.assertTrue(s.model.paused)
        self.assertTrue(s.model.is_initial())
        s.resume()

    # ------------------------------------------------------------ RESTART
    def test_restart_reproduces_initial_simulation(self):
        s = self.sim
        s.restart()
        s.model.set_control(0.9)
        s.model.warp = 86400.0
        s.pause()
        self._frames(10)
        s.restart()
        self.assertTrue(s.model.is_initial())
        self.assertEqual(s.model.control, 0.6)                        # run's initial control
        self.assertEqual(s.model.warp, 3600.0)                        # run's initial warp
        self.assertFalse(s.model.paused)

    def test_restart_same_seed_bit_identical_visuals_and_physics(self):
        s = self.sim
        s.restart()
        self._frames(25)
        phys_a, vis_a = physics_snapshot(s.model), visual_snapshot(s)
        s.model.set_control(0.1)                                      # perturb, then restart
        self._frames(7)
        s.restart()
        self._frames(25)
        phys_b, vis_b = physics_snapshot(s.model), visual_snapshot(s)
        self.assertEqual(phys_a, phys_b)
        for k in vis_a:
            self.assertTrue(np.array_equal(vis_a[k], vis_b[k]), f"visual buffer {k} differs after restart")

    def test_different_seed_gives_different_visuals_same_physics(self):
        s = self.sim
        s.restart()
        self._frames(10)
        phys_a, vis_a = physics_snapshot(s.model), visual_snapshot(s)
        s.renderer.reseed(8)
        s.model.reset()
        s._apply_initial_state()
        self._frames(10)
        self.assertEqual(physics_snapshot(s.model), phys_a)           # physics has no seed dependence
        self.assertFalse(np.array_equal(vis_a["stars"], visual_snapshot(s)["stars"]))
        self.assertFalse(np.array_equal(vis_a["wind_dir"], visual_snapshot(s)["wind_dir"]))
        s.restart()

    def test_keyboard_hooks_exist(self):
        """The key map wires SPACE/R/T/X/F/LEFT/RIGHT to the hooks (source-level guard)."""
        import simulation
        src = open(simulation.__file__, encoding="utf-8").read()
        for token in ("ti.ui.SPACE", '"r"', '"t"', '"x"', '"f"', "ti.ui.LEFT", "ti.ui.RIGHT", "toggle_pause", "self.reset()", "self.restart()"):
            self.assertIn(token, src)


if __name__ == "__main__":
    unittest.main()
