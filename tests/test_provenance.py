"""Provenance registry: observed vs computed vs simplified is machine-readable and consistent."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                  # noqa: E402
from physics import provenance as pv                  # noqa: E402
from physics.model import SystemModel                 # noqa: E402


class TestProvenance(unittest.TestCase):
    def test_types_are_explicit(self):
        types = {e[1] for e in pv.INPUTS}
        self.assertEqual(types, {pv.OBSERVED, pv.OBSERVED_DERIVED, pv.SIMPLIFIED})
        for e in pv.observed_inputs():
            self.assertRegex(e[5], r"\[[A-Z]\d\d\]", f"missing citation for {e[0]}")
        for e in pv.simplified_inputs():
            self.assertNotIn("[", e[5], f"simplified entry {e[0]} must not carry an observation citation")

    def test_observed_values_match_config(self):
        by_attr = {e[2]: e for e in pv.INPUTS if e[2]}
        self.assertAlmostEqual(by_attr["NS_SPIN_PERIOD_OBS"][3], cfg.NS_SPIN_PERIOD_OBS * 1e3)
        self.assertAlmostEqual(by_attr["NS_MASS_OBS"][3], cfg.NS_MASS_OBS / cfg.M_SUN)
        self.assertAlmostEqual(by_attr["B_SURFACE_OBS"][3], cfg.B_SURFACE_OBS / cfg.GAUSS)

    def test_every_telemetry_key_classified(self):
        m = SystemModel()
        m.control = 0.5
        m.advance(cfg.HOUR)
        for k in m.telemetry():
            self.assertIsNotNone(pv.classify(k), f"unclassified telemetry key {k}")
            self.assertIn(pv.classify(k), (pv.COMPUTED, "USER SETTING"))

    def test_table_and_readme(self):
        rows = pv.table_rows()
        self.assertTrue(all(len(r) == 4 and r[3] for r in rows))
        md = pv.format_table()
        self.assertTrue(md.startswith("| Quantity | Type | Source / Model | Units |"))
        readme = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"), encoding="utf-8").read()
        for line in md.splitlines()[:10]:
            self.assertIn(line, readme)


if __name__ == "__main__":
    unittest.main()
