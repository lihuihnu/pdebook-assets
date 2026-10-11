"""ET005-01 small-grid/analytic gates, independent of manuscript figures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from burgers_flux import numerical_flux, physical
from burgers_fvm import face_fluxes, euler_step, periodic_initial_means, riemann_initial_means, total_variation
from exact_references import BREAK_TIME, smooth_reference, invert_smooth_characteristics
from run_ET005_01 import run, sha_file
from validate_ET005_01 import audit


class BurgersContractTests(unittest.TestCase):
    def test_registered_flux_values(self):
        self.assertAlmostEqual(float(numerical_flux(1., 0.)), .5)
        self.assertAlmostEqual(float(numerical_flux(-1., 1.)), 0.)
        self.assertAlmostEqual(float(numerical_flux(-1., 1., "Roe_no_entropy_fix")), .5)
        self.assertAlmostEqual(float(numerical_flux(1., 0., "LF_alpha1")), .75)
        for method in ("Godunov", "Roe_no_entropy_fix", "LF_alpha1"):
            for c in (-1., -.33, 0., .29, 1.):
                self.assertAlmostEqual(float(numerical_flux(c, c, method)), .5*c*c)
        with self.assertRaises(ValueError):
            numerical_flux(0., 1., "unknown")

    def test_periodic_two_cells_hand_assembly(self):
        old = np.array([.1, .8])
        out, face = euler_step(old, .5, .1, "periodic")
        self.assertEqual(len(face), 3)
        self.assertEqual(face[0], face[-1])
        self.assertAlmostEqual(face[0], .5*.8**2)
        self.assertAlmostEqual(face[1], .5*.1**2)
        expected = np.array([.1-.2*(.005-.32), .8-.2*(.32-.005)])
        np.testing.assert_allclose(out, expected, rtol=0, atol=2e-16)
        self.assertAlmostEqual(np.sum(out), np.sum(old))

    def test_periodic_three_cells_and_constant(self):
        a = np.array([.75,.5,.25])
        f = face_fluxes(a, "periodic")
        for i in range(3):
            self.assertAlmostEqual(f[i], float(numerical_flux(a[(i-1)%3],a[i])))
        for c in (-.75, 0.0, .6):
            u = np.full(3,c)
            out, faces = euler_step(u, 1./3., .1, "periodic")
            np.testing.assert_array_equal(u,out)
            self.assertAlmostEqual(total_variation(out,"periodic"),0.)

    def test_riemann_initial_average_and_boundary_flux(self):
        u = riemann_initial_means(4, 1., 0.)
        np.testing.assert_array_equal(u, [1.,1.,0.,0.])
        f = face_fluxes(u, "riemann",left=1.,right=0.)
        self.assertAlmostEqual(f[0],.5)
        self.assertAlmostEqual(f[-1],0.)
        out,_ = euler_step(u,.5,.4,"riemann",left=1.,right=0.)
        self.assertAlmostEqual(.5*sum(out) - .5*sum(u), .4*(.5-0.))

    def test_cfl_rejects_before_any_computation(self):
        with self.assertRaisesRegex(ValueError, "CFL"):
            euler_step(np.array([.2,.4,.5]), 1./3, 1.1/3, "periodic")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            numerical_flux(np.nan,1.)

    def test_sine_initial_cell_integral(self):
        u=periodic_initial_means(200)
        self.assertLess(abs(np.mean(u)-.5),1e-15)
        self.assertGreater(np.max(np.abs(u-(.5+.25*np.sin(2*np.pi*(np.arange(200)+.5)/200)))),1e-7)

    def test_smooth_characteristics_and_periodic_lift(self):
        self.assertAlmostEqual(BREAK_TIME,2/np.pi)
        self.assertAlmostEqual(.5 + .5*BREAK_TIME,.5+1/np.pi)
        for t in (0.,.3,.6):
            ref=smooth_reference(200,t)
            self.assertLess(ref["max_root_residual"],2e-13)
            self.assertLess(ref["periodic_xi_shift_error"],2e-13)
            self.assertLess(ref["max_quad_disagreement"],2e-10)
            self.assertLess(abs(ref["exact"].mean()-.5),1e-13)
        xi,resid=invert_smooth_characteristics(np.array([0.,1.]),.6)
        self.assertLess(resid,2e-13)
        self.assertAlmostEqual(xi[1]-xi[0],1.)
        with self.assertRaises(ValueError):
            smooth_reference(200,BREAK_TIME)
        with self.assertRaises(ValueError):
            smooth_reference(200,.7)

    def test_end_to_end_recompute_and_independent_validation(self):
        c=ROOT/"config/design_v01.json"
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/"A",Path(temp)/"B"
            run(c,a,"test-placeholder-sha")
            run(c,b,"test-placeholder-sha")
            for filename in ("snapshots.csv","metrics.csv","step_budget.csv","reference_cellavg.csv"):
                self.assertEqual(sha_file(a/filename),sha_file(b/filename))
            result=audit(a,c)
            self.assertEqual(result["status"],"PASS")
            self.assertEqual(result["actual_steps"],200)
            self.assertEqual(result["snapshots"],1000)
            self.assertEqual(result["reference_rows"],600)
            self.assertLess(result["max_mass_defect"],1e-11)


if __name__ == "__main__":
    unittest.main()
