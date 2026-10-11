"""ET005-02 shock, exact cell means, fixed exterior boundary and tamper gates."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from burgers_fvm import riemann_initial_means, face_fluxes, euler_step
from exact_references import exact_shock_cell_means
from run_ET005_02 import run, unique_half_crossing, sha256, registered_config
from validate_ET005_02 import audit, independent_exact_means, scalar_godunov


class ShockExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config=ROOT/"config/ET005-02.json"
        self.design=ROOT/"config/design_v01.json"

    def test_frozen_contract_and_flux(self):
        cfg=registered_config(self.config,self.design)
        self.assertEqual(cfg["times"],[0,.1,.25,.5])
        self.assertEqual(cfg["N"],200)
        self.assertEqual(scalar_godunov(1.,0.),.5)
        self.assertEqual(scalar_godunov(0.,0.),0.)
        self.assertEqual(scalar_godunov(-1.,1.),0.)
        self.assertAlmostEqual(scalar_godunov(.2,.2),.02)

    def test_exact_initial_integral_and_fractional_shock_cell(self):
        a=riemann_initial_means(4,1.,0.)
        np.testing.assert_array_equal(a,[1.,1.,0.,0.])
        ref=exact_shock_cell_means(4,.5)
        np.testing.assert_array_equal(ref,[1.,1.,.5,0.])
        for time in (0.,.1,.25,.5):
            reference=exact_shock_cell_means(200,time)
            np.testing.assert_allclose(reference,independent_exact_means(time),atol=3e-14,rtol=0)
            self.assertAlmostEqual(float(reference.sum())*.01,1.+time*.5,places=13)

    def test_nonperiodic_single_face_and_fixed_exterior(self):
        u=np.array([1.,1.,0.,0.])
        faces=face_fluxes(u,"riemann",left=1.,right=0.)
        np.testing.assert_array_equal(faces,[.5,.5,.5,0.,0.])
        nxt,_=euler_step(u,.5,.4,"riemann",left=1.,right=0.)
        np.testing.assert_allclose(nxt,[1.,1.,.4,0.],atol=1e-15,rtol=0)
        self.assertAlmostEqual(.5*sum(nxt)-.5*sum(u),.4*.5)

    def test_half_level_location_independent_of_mass(self):
        self.assertAlmostEqual(unique_half_crossing(np.array([1.,1.,0.,0.]),.5),0.)
        with self.assertRaisesRegex(ValueError,"unique"):
            unique_half_crossing(np.array([1.,0.,1.,0.]),.5)
        with self.assertRaisesRegex(ValueError,"unique"):
            unique_half_crossing(np.array([0.,0.,0.,0.]),.5)

    def test_actual_computation_independent_replay_and_deterministic_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,second=Path(tmp)/"A",Path(tmp)/"B"
            run(self.config,self.design,first,"test-experiment-source")
            run(self.config,self.design,second,"test-experiment-source")
            for name in ("snapshots.csv","metrics.csv","step_budget.csv"):
                self.assertEqual(sha256(first/name),sha256(second/name))
            verified=audit(first,self.config,self.design)
            self.assertEqual(verified["status"],"PASS")
            self.assertAlmostEqual(verified["analytic_mass_T"],1.25)
            self.assertEqual(verified["analytic_shock_x_T"],.25)
            self.assertLess(verified["max_step_and_cumulative_mass_defect"],1e-11)
            with (first/"metrics.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([float(r["time"]) for r in rows],[0,.1,.25,.5])
            self.assertEqual([int(r["steps"]) for r in rows],[0,13,32,64])
            self.assertAlmostEqual(float(rows[-1]["mass"]),1.25,places=11)
            # Corrupt an independently verified boundary flux without updating hash.
            ledger=first/"step_budget.csv"
            ledger.write_text(ledger.read_text().replace("F_left", "FL_wrong",1))
            with self.assertRaisesRegex(ValueError,"SHA"):
                audit(first,self.config,self.design)


if __name__=="__main__":
    unittest.main()
