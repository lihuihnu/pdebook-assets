"""ET005-05 registered LF/Godunov 2x3 verification and reject-on-tamper tests."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from burgers_fvm import euler_step, face_fluxes, riemann_initial_means
from burgers_flux import numerical_flux
from exact_references import exact_shock_cell_means
from run_ET005_05 import (
    registered_config, run, sha, transition_region, quadratic_content
)
from validate_ET005_05 import (
    audit, scalar_flux, integral_exact, original_means,
    shape_diagnostics, FROZEN_ET00502_HASHES
)


class CFLNumericalDissipationTests(unittest.TestCase):
    def setUp(self):
        self.config=ROOT/"config/ET005-05.json"
        self.design=ROOT/"config/design_v01.json"
        self.archive=ROOT/"results"

    def test_complete_frozen_2_by_3_design(self):
        cfg=registered_config(self.config,self.design)
        self.assertEqual(cfg["methods"],["Godunov","LF_alpha1"])
        self.assertEqual(cfg["requested_cfl"],[0.4,0.7,0.9])
        self.assertEqual(cfg["N"],200)
        self.assertEqual(cfg["times"],[0,0.1,0.25,0.5])
        self.assertEqual(cfg["LF_alpha"],1.0)

    def test_flux_is_truly_global_alpha_one(self):
        self.assertEqual(float(numerical_flux(1.,0.,"Godunov")),.5)
        self.assertEqual(float(numerical_flux(1.,0.,"LF_alpha1")),.75)
        self.assertEqual(scalar_flux(1.,0.,"LF_alpha1"),.75)
        for a,b in ((.2,.3),(.8,.1),(-1.,1.),(.1,.1)):
            expected=.25*(a*a+b*b)-.5*(b-a)
            self.assertAlmostEqual(scalar_flux(a,b,"LF_alpha1"),expected)
            self.assertAlmostEqual(float(numerical_flux(a,b,"LF_alpha1")),expected)
        self.assertAlmostEqual(float(numerical_flux(.3,.3,"LF_alpha1")),.045)

    def test_true_cell_integral_initial_and_reference(self):
        raw=original_means()
        exact=integral_exact()
        np.testing.assert_allclose(raw,riemann_initial_means(200,1.,0.),
                                   atol=2e-14,rtol=0)
        np.testing.assert_allclose(exact,exact_shock_cell_means(200,.5),
                                   atol=4e-14,rtol=0)
        self.assertAlmostEqual(float(exact.sum())*.01,1.25,places=12)
        np.testing.assert_allclose(integral_exact(n=4),[1.,1.,.5,0.],
                                   atol=2e-15,rtol=0)

    def test_two_fluxes_same_shared_face_update(self):
        raw=np.array([1.,1.,0.,0.])
        gf=face_fluxes(raw,"riemann","Godunov",left=1.,right=0.)
        lf=face_fluxes(raw,"riemann","LF_alpha1",left=1.,right=0.)
        np.testing.assert_allclose(gf,[.5,.5,.5,0,0],atol=2e-15,rtol=0)
        np.testing.assert_allclose(lf,[.5,.5,.75,0,0],atol=2e-15,rtol=0)
        gnext,_=euler_step(raw,.5,.4,"riemann","Godunov",left=1.,right=0.)
        lnext,_=euler_step(raw,.5,.4,"riemann","LF_alpha1",left=1.,right=0.)
        np.testing.assert_allclose(gnext,[1.,1.,.4,0.],atol=2e-15,rtol=0)
        np.testing.assert_allclose(lnext,[1.,.8,.6,0.],atol=2e-15,rtol=0)
        self.assertAlmostEqual(.5*sum(lnext),1.2,places=14)

    def test_transition_width_strict_and_disconnected_segments(self):
        u=np.zeros(200)
        u[4:7]=[.1,.2,.3]
        u[18:21]=[.7,.9,.8]
        cells,segments,width=transition_region(u)
        self.assertEqual((cells,segments),(4,3))
        self.assertAlmostEqual(width,.04)
        self.assertEqual(transition_region(np.zeros(200)),(0,0,0.))
        self.assertAlmostEqual(quadratic_content(np.ones(200)),1.)
        with self.assertRaises(ValueError):
            transition_region(np.full(200,np.nan))

    def test_cfl_1p1_is_rejected_without_running_unstable_control(self):
        raw=np.array([1.,1.,0.,0.],dtype=np.float64)
        for method in ("Godunov","LF_alpha1"):
            with self.assertRaisesRegex(ValueError,"CFL"):
                euler_step(raw,.5,.55,"riemann",method,left=1.,right=0.)
        np.testing.assert_array_equal(raw,[1.,1.,0.,0.])

    def test_frozen_historical_shock_data_identity(self):
        for name,digest in FROZEN_ET00502_HASHES.items():
            self.assertEqual(sha(self.archive/"ET005-02"/name),digest)

    def test_six_actual_runs_twice_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"A",Path(tmp)/"B"
            run(self.config,self.design,a,"unit-test-source")
            run(self.config,self.design,b,"unit-test-source")
            for name in ("cfl_flux_metrics.csv","snapshots.csv","step_budget.csv"):
                self.assertEqual(sha(a/name),sha(b/name))
            check=audit(a,self.config,self.design,self.archive)
            self.assertEqual(check["status"],"PASS")
            self.assertEqual(check["metric_rows"],6)
            self.assertEqual(check["snapshot_rows"],1200)
            self.assertEqual(len(check["verified_pairs"]),6)
            self.assertLess(check["max_mass_budget_defect"],1e-11)
            with (a/"cfl_flux_metrics.csv").open(newline="") as f:
                metrics=list(csv.DictReader(f))
            self.assertEqual([(r["method"],float(r["cfl_requested"])) for r in metrics],
                [(m,c) for m in ("Godunov","LF_alpha1") for c in (.4,.7,.9)])
            self.assertTrue(all(0<=int(r["transition_cells"])<=200 for r in metrics))
            # Any tampering must be detected independently even if it looks plausible.
            csv_path=a/"step_budget.csv"
            csv_path.write_text(csv_path.read_text(encoding="utf-8")
                                .replace("F_left","F_untrusted",1),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                audit(a,self.config,self.design,self.archive)


if __name__=="__main__":
    unittest.main()
