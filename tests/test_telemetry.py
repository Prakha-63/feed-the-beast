"""The HUD telemetry text is built from live model values with units."""
import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                                   # noqa: E402
from physics.model import SystemModel                                  # noqa: E402
from render.telemetry_text import telemetry_lines, control_lines, fmt_sci   # noqa: E402

REQUIRED = {
    "MASS": "Msun", "ACCRETION": "Msun/yr", "ANG. MOM.": "kg m^2/s", "ROTATION": "Hz", "PERIOD": "ms",
    "MAG. FIELD": " G", "ROT. ENERGY": " J", "ACTIVITY": "erg/s", "SIM TIME": "1 s =", "STATE": "",
}


class TestFormatting(unittest.TestCase):
    def test_fmt_sci(self):
        self.assertEqual(fmt_sci(0.0, "kg"), "0 kg")
        self.assertEqual(fmt_sci(1234.5, "kg"), "1.23e+3 kg")
        self.assertEqual(fmt_sci(-3.2e-12, "Msun/yr"), "-3.20e-12 Msun/yr")
        self.assertEqual(fmt_sci(9.9999e5, "J"), "1.00e+6 J")               # rounding overflow handled
        self.assertEqual(fmt_sci(float("inf"), "km"), "-- km")
        self.assertEqual(fmt_sci(7.29e41, "kg m^2/s", 4), "7.290e+41 kg m^2/s")


class TestTelemetryLines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = SystemModel()
        cls.m.control = 0.8
        for _ in range(80):
            cls.m.advance(cfg.HOUR)
        cls.t = cls.m.telemetry()
        cls.lines = [txt for txt, _ in telemetry_lines(cls.t, cls.m.warp_label, cls.m.kin_capped)]

    def _line(self, label):
        for ln in self.lines:
            if ln.startswith(label):
                return ln
        self.fail(f"no telemetry line for {label}")

    def test_required_quantities_with_units(self):
        for label, unit in REQUIRED.items():
            ln = self._line(label)
            self.assertIn(unit, ln, f"{label} line lacks unit {unit!r}: {ln}")

    def test_values_come_from_the_model(self):
        t, m = self.t, self.m
        self.assertIn(f"{m.ns.mass / cfg.M_SUN:.6f} Msun", self._line("MASS"))
        self.assertIn(f"{m.ns.frequency:.4f} Hz", self._line("ROTATION"))
        self.assertIn(f"{m.ns.period * 1e3:.6f} ms", self._line("PERIOD"))
        self.assertIn(fmt_sci(m.ns.J, "kg m^2/s"), self._line("ANG. MOM."))
        self.assertIn(fmt_sci(m.ns.B / cfg.GAUSS, "G"), self._line("MAG. FIELD"))
        self.assertIn(fmt_sci(m.ns.rotational_energy, "J"), self._line("ROT. ENERGY"))
        self.assertIn(f"{m.activity:.3f}", self._line("ACTIVITY"))
        self.assertIn(m.state, self._line("STATE"))
        self.assertIn(m.clock.formatted(), self._line("SIM TIME"))

    def test_lines_change_with_the_physics(self):
        m2 = SystemModel()
        m2.control = 0.0
        m2.advance(cfg.DAY)
        other = [txt for txt, _ in telemetry_lines(m2.telemetry(), m2.warp_label, m2.kin_capped)]
        for label in ("STATE", "ACCRETION", "ACTIVITY", "SIM TIME", "DISK MASS"):
            a = self._line(label)
            b = next(x for x in other if x.startswith(label))
            self.assertNotEqual(a, b, label)

    def test_precision_is_visible(self):
        """Tiny physical changes must still show: accreted mass and delta-f in scientific notation."""
        ln = self._line("MASS")
        self.assertRegex(ln, r"accreted \d\.\d\de[+-]\d+ Msun")
        self.assertRegex(self._line("ROTATION"), r"\([+-]\d\.\d{3}e[+-]\d+ Hz since start\)")

    def test_nan_inf_safe(self):
        z = SystemModel()
        lines = telemetry_lines(z.telemetry(), z.warp_label, False)         # r_m = inf, fastness = inf at start
        joined = "\n".join(txt for txt, _ in lines)
        self.assertNotIn("nan", joined.lower())
        self.assertNotIn("inf ", joined.lower())

    def test_control_lines(self):
        c = control_lines(self.t, self.m.warp_label, "FREE ORBIT", "VULKAN", 42.0)
        self.assertIn("Msun/yr", c[0])
        self.assertIn(self.m.warp_label, c[1])
        self.assertIn("42 fps", c[-1])


if __name__ == "__main__":
    unittest.main()
