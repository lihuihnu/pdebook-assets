"""Additional ET007-02 regression tests; do not change frozen ET007-01 code."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.model import assemble, dual_face_flux, nodal_solution, tensor, report_metrics


class AffineRotatedDirichletTests(unittest.TestCase):
    def test_angle_45_tensor_is_symmetric_positive_definite(self):
        a,b,c=tensor(45.)
        np.testing.assert_allclose([a,b,c],[5.5,4.5,5.5],rtol=0,atol=2e-14)
        vals=np.linalg.eigvalsh([[a,b],[b,c]])
        np.testing.assert_allclose(vals,[1.,10.],rtol=0,atol=2e-14)

    def test_both_actual_sparse_solves_recover_nonzero_dirichlet_affine(self):
        n=8
        xx=np.arange(n+1,dtype=float)[:,None]/n
        for method in ("M0","M1"):
            with self.subTest(method=method):
                problem=assemble(n,45.,method,"affine")
                u,residual=nodal_solution(problem)
                self.assertLess(residual,1e-11)
                np.testing.assert_allclose(u,np.broadcast_to(xx,(n+1,n+1)),
                                           rtol=0,atol=1e-12)
                self.assertGreater(problem.A.nnz,0)

    def test_missing_cross_flux_has_exactly_forty_five_tenths_error(self):
        n=8
        for method in ("M0","M1"):
            with self.subTest(method=method):
                u,_=nodal_solution(assemble(n,45.,method,"affine"))
                qx,qy=dual_face_flux(u,45.,method)
                measured=report_metrics(u,45.,method,"affine")
                np.testing.assert_allclose(qx,np.full_like(qx,-5.5),
                                           atol=8e-12,rtol=0)
                self.assertLess(measured["T_error_linf"],1e-12)
                if method=="M0":
                    np.testing.assert_allclose(qy,np.zeros_like(qy),
                                               atol=8e-12,rtol=0)
                    self.assertAlmostEqual(measured["q_y_error_linf"],4.5,places=11)
                else:
                    np.testing.assert_allclose(qy,np.full_like(qy,-4.5),
                                               atol=8e-12,rtol=0)
                    self.assertLess(measured["q_y_error_linf"],1e-11)
                self.assertLess(measured["dual_cell_balance_scaled"],1e-11)

    def test_internal_dual_budget_is_not_physical_boundary_flux(self):
        n=8
        for method in ("M0","M1"):
            u,_=nodal_solution(assemble(n,45.,method,"affine"))
            observed=report_metrics(u,45.,method,"affine")
            self.assertLess(observed["dual_boundary_budget_scaled"],1e-11)
        # The M0 normal component qy on horizontal faces is wrong despite
        # having a negligible balance on its *own* Omega_h dual cells.
        u,_=nodal_solution(assemble(n,45.,"M0","affine"))
        _,qy=dual_face_flux(u,45.,"M0")
        self.assertLess(float(np.max(np.abs(qy))),1e-11)

    def test_analytic_rotation_sign_negative_45(self):
        a,b,c=tensor(-45.)
        self.assertAlmostEqual(-b,4.5,places=12)
        self.assertAlmostEqual(-a,-5.5,places=12)
        self.assertAlmostEqual(c,5.5,places=12)


if __name__=="__main__":
    unittest.main()
