"""ET005-03 tests: exact fan means, entropy pathology, replay and corruption gate."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burgers_flux import numerical_flux
from burgers_fvm import euler_step, face_fluxes, riemann_initial_means
from exact_rarefaction import entropy_rarefaction_cell_means
from run_ET005_03 import registered_config, run, sha256
from validate_ET005_03 import audit, independent_integral_reference, scalar_flux


class EntropyCounterexampleTests(unittest.TestCase):
    def setUp(self):
        self.config = ROOT / "config/ET005-03.json"
        self.design = ROOT / "config/design_v01.json"

    def test_frozen_design_and_transonic_faces(self):
        cfg = registered_config(self.config, self.design)
        self.assertEqual(cfg["methods"], ["Godunov", "Roe_no_entropy_fix"])
        self.assertEqual(cfg["times"], [0., .1, .25, .5])
        self.assertEqual(float(numerical_flux(-1, 1, "Godunov")), 0.)
        self.assertEqual(float(numerical_flux(-1, 1, "Roe_no_entropy_fix")), .5)
        self.assertEqual(scalar_flux(-1., 1., "Roe_no_entropy_fix"), .5)
        self.assertAlmostEqual((1.**3 - (-1.)**3)/3., 2./3.)

    def test_exact_rarefaction_means_not_center_value(self):
        for n in (4, 200):
            for t in (0., .1, .25, .5):
                ref = entropy_rarefaction_cell_means(n, t)
                self.assertTrue(np.isfinite(ref).all())
                self.assertAlmostEqual(float(ref.sum()) * 2. / n, 0., places=12)
                if n == 200:
                    np.testing.assert_allclose(
                        ref, independent_integral_reference(t), rtol=0, atol=2e-13)
        np.testing.assert_allclose(entropy_rarefaction_cell_means(4, .5),
                                   [-1., -.5, .5, 1.], rtol=0, atol=2e-15)

    def test_same_initial_means_and_uncorrected_roe_static_step(self):
        u = riemann_initial_means(4, -1., 1.)
        np.testing.assert_allclose(u, [-1., -1., 1., 1.], rtol=0, atol=2e-15)
        roe_faces = face_fluxes(u, "riemann", method="Roe_no_entropy_fix", left=-1., right=1.)
        np.testing.assert_allclose(roe_faces, [.5]*5, rtol=0, atol=2e-15)
        roe_next, _ = euler_step(u, .5, .4, "riemann", method="Roe_no_entropy_fix",
                                 left=-1., right=1.)
        np.testing.assert_allclose(roe_next, u, rtol=0, atol=2e-15)
        godunov_next, gf = euler_step(u, .5, .4, "riemann", method="Godunov",
                                     left=-1., right=1.)
        self.assertEqual(float(gf[2]), 0.)
        np.testing.assert_allclose(godunov_next, [-1., -.6, .6, 1.],
                                   rtol=0, atol=2e-15)

    def test_fail_closed_on_bad_cfl(self):
        u = np.array([-1., -1., 1., 1.])
        with self.assertRaisesRegex(ValueError, "CFL"):
            euler_step(u, .5, .55, "riemann", "Godunov", left=-1., right=1.)

    def test_actual_two_methods_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d)/"A", Path(d)/"B"
            for path in (first, second):
                run(self.config, self.design, path, "unit-test-source")
            for name in ("snapshots.csv", "metrics.csv", "step_budget.csv"):
                self.assertEqual(sha256(first/name), sha256(second/name))
            result = audit(first, self.config, self.design)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["steps_per_method"], 64)
            self.assertEqual(result["snapshot_rows"], 1600)
            self.assertEqual(result["step_budget_rows"], 128)
            self.assertLess(result["T0_5"]["Godunov"]["L1_cellavg_T"], .5)
            self.assertAlmostEqual(result["T0_5"]["Roe_no_entropy_fix"]["L1_cellavg_T"],
                                   .5, places=11)
            self.assertAlmostEqual(result["T0_5"]["Roe_no_entropy_fix"]["entropy_balance_T"],
                                   1./3., places=11)
            with (first/"metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["method"] for r in rows[:4]], ["Godunov"]*4)
            self.assertEqual([r["method"] for r in rows[4:]], ["Roe_no_entropy_fix"]*4)
            ledger = first/"step_budget.csv"
            ledger.write_text(ledger.read_text().replace("F_left", "fake_left", 1))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit(first, self.config, self.design)


if __name__ == "__main__":
    unittest.main()
