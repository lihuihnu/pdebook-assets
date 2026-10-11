"""ET005-04: registered grid, exact integrals, conservation and frozen crosschecks."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from burgers_fvm import riemann_initial_means, euler_step
from exact_references import exact_shock_cell_means
from exact_rarefaction import entropy_rarefaction_cell_means
from run_ET005_04 import frozen_config, run, sha256
from validate_ET005_04 import (
    audit, reference_integral, godunov_scalar, FROZEN_ARCHIVE_SHA256
)


class GodunovRefinementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ROOT/"config/ET005-04.json"
        self.design = ROOT/"config/design_v01.json"
        self.archive_root = ROOT/"results"

    def test_frozen_complete_two_by_four_matrix(self):
        cfg=frozen_config(self.cfg,self.design)
        self.assertEqual(cfg["cases"],["shock_1_0","rarefaction_m1_1"])
        self.assertEqual(cfg["N"],[100,200,400,800])
        self.assertEqual(cfg["times"],[0.,.1,.25,.5])
        self.assertEqual(cfg["requested_cfl"],.8)
        self.assertEqual(cfg["method"],"Godunov")
        self.assertEqual(godunov_scalar(1.,0.),.5)
        self.assertEqual(godunov_scalar(-1.,1.),0.)

    def test_shock_exact_means_and_mass(self):
        for n in (100,200,400,800):
            actual=exact_shock_cell_means(n,.5)
            independent=reference_integral("shock_1_0",n,.5)
            np.testing.assert_allclose(actual,independent,atol=3e-13,rtol=0)
            self.assertAlmostEqual(float(actual.sum())*(2./n),1.25,places=12)

    def test_fan_exact_means_and_mass(self):
        for n in (100,200,400,800):
            actual=entropy_rarefaction_cell_means(n,.5)
            independent=reference_integral("rarefaction_m1_1",n,.5)
            np.testing.assert_allclose(actual,independent,atol=3e-13,rtol=0)
            self.assertAlmostEqual(float(actual.sum())*(2./n),0.,places=12)
        np.testing.assert_allclose(reference_integral("rarefaction_m1_1",4),
                                   [-1.,-.5,.5,1.],atol=1e-14,rtol=0)

    def test_no_implicit_cfl_or_state_clip(self):
        state=riemann_initial_means(4,1.,0.)
        with self.assertRaisesRegex(ValueError,"CFL"):
            euler_step(state,.5,.55,"riemann",method="Godunov",left=1.,right=0.)
        next_state,faces=euler_step(state,.5,.4,"riemann",
                                    method="Godunov",left=1.,right=0.)
        np.testing.assert_allclose(faces,[.5,.5,.5,0.,0.],atol=2e-15,rtol=0)
        np.testing.assert_allclose(next_state,[1.,1.,.4,0.],atol=2e-15,rtol=0)

    def test_existing_N200_archives_are_sha_pinned(self):
        for case,eid in (("shock_1_0","ET005-02"),
                         ("rarefaction_m1_1","ET005-03")):
            for name,expected in FROZEN_ARCHIVE_SHA256[case].items():
                self.assertEqual(sha256(self.archive_root/eid/name),expected)

    def test_8_actual_runs_double_determinism_and_strict_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"A",Path(directory)/"B"
            run(self.cfg,self.design,a,"test-source-id")
            run(self.cfg,self.design,b,"test-source-id")
            for name in ("grid_errors.csv","snapshots.csv","step_budget.csv"):
                self.assertEqual(sha256(a/name),sha256(b/name))
            check=audit(a,self.cfg,self.design,self.archive_root)
            self.assertEqual(check["status"],"PASS")
            self.assertEqual(check["grid_rows"],8)
            self.assertEqual(check["snapshot_rows"],3000)
            self.assertEqual(set(check["N200_frozen_crosschecks"]),
                             {"shock_1_0","rarefaction_m1_1"})
            self.assertLess(check["max_step_and_cumulative_mass_defect"],1e-11)
            with (a/"grid_errors.csv").open(newline="",encoding="utf-8") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([(x["case_id"],int(x["N"])) for x in rows],
                             [(case,n) for case in ("shock_1_0","rarefaction_m1_1")
                              for n in (100,200,400,800)])
            self.assertTrue(all(float(x["L1_cellavg"])>=0. for x in rows))
            self.assertTrue(all(x["p_obs_from_previous"]=="" for x in
                                (rows[0],rows[4])))
            ledger=a/"step_budget.csv"
            ledger.write_text(ledger.read_text().replace("F_left","fake_left",1),
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                audit(a,self.cfg,self.design,self.archive_root)


if __name__=="__main__":
    unittest.main()
