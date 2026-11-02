"""Small algebraic ET007-03 regressions; these are NOT eight production runs."""
from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import (assemble, exact_and_source, dual_face_flux, nodal_solution,
                       tensor)
from src.validate_et007_03 import (analytic_values, independent_operator,
                                   matrix_free_M1, spectral_M0)


class ManufacturedSourceTests(unittest.TestCase):
    def test_rotation_tensor_and_full_mixed_source(self):
        kxx,kxy,kyy=tensor(45)
        np.testing.assert_allclose([kxx,kxy,kyy],[5.5,4.5,5.5],
                                   atol=3e-15,rtol=0)
        x,y=np.array(.125),np.array(.375)
        exact,source=exact_and_source("sinesine",x,y,kxx,kxy,kyy)
        expression=(11*math.pi**2*math.sin(math.pi*.125)*math.sin(math.pi*.375)
                    -9*math.pi**2*math.cos(math.pi*.125)*math.cos(math.pi*.375))
        self.assertAlmostEqual(float(source),expression,places=12)
        self.assertAlmostEqual(float(exact),
                               math.sin(math.pi*.125)*math.sin(math.pi*.375),places=15)

    def test_the_full_source_is_identical_in_both_methods(self):
        for n in (4,8,16):
            m0=assemble(n,45,"M0","sinesine")
            m1=assemble(n,45,"M1","sinesine")
            with self.subTest(n=n):
                self.assertTrue(np.array_equal(m0.b,m1.b))
                self.assertGreater(m1.A.nnz,m0.A.nnz)
                self.assertEqual(m0.A.nnz,5*(n-1)**2-4*(n-1))
                self.assertEqual(m1.A.nnz,9*(n-1)**2-12*(n-1)+4)

    def test_ninepoint_diagonal_couplings_have_the_right_sign(self):
        n=8
        m1=assemble(n,45,"M1","sinesine")
        i,j=4,4
        idx=lambda a,b:(a-1)*(n-1)+(b-1)
        row=idx(i,j)
        delta=4.5*n*n/2
        self.assertAlmostEqual(float(m1.A[row,idx(i+1,j+1)]),-delta)
        self.assertAlmostEqual(float(m1.A[row,idx(i+1,j-1)]),+delta)
        self.assertAlmostEqual(float(m1.A[row,idx(i-1,j+1)]),+delta)
        self.assertAlmostEqual(float(m1.A[row,idx(i-1,j-1)]),-delta)

    def test_zero_dirichlet_values_are_exact_at_the_sine_boundary(self):
        n=8
        xx,yy=np.meshgrid(np.arange(n+1)/n,np.arange(n+1)/n,indexing="ij")
        t,_=exact_and_source("sinesine",xx,yy,5.5,4.5,5.5)
        self.assertTrue(np.array_equal(t[0,:],np.zeros(n+1)))
        self.assertTrue(np.array_equal(t[n,:],np.zeros(n+1)))
        self.assertTrue(np.array_equal(t[:,0],np.zeros(n+1)))
        self.assertTrue(np.array_equal(t[:,n],np.zeros(n+1)))

    def test_independent_operator_agrees_with_flux_divergence_of_xy(self):
        n=8
        h=1/n
        x=np.arange(n+1)*h
        u=x[:,None]*x[None,:]
        qx,qy=dual_face_flux(u,45,"M1")
        divergence=(qx[1:]-qx[:-1]+qy[:,1:]-qy[:,:-1])/h
        np.testing.assert_allclose(divergence,np.full((n-1,n-1),-9.0),
                                   atol=2e-12,rtol=0)
        # The matrix-free routine assumes zero Dirichlet data. With explicit
        # true boundary terms subtracted, it must equal the direct divergence.
        inside=u[1:-1,1:-1].ravel()
        op=independent_operator(inside,n,"M1").reshape(n-1,n-1)
        boundary_only=np.array(u)
        boundary_only[1:-1,1:-1]=0.
        qbx,qby=dual_face_flux(boundary_only,45,"M1")
        boundary_div=(qbx[1:]-qbx[:-1]+qby[:,1:]-qby[:,:-1])/h
        np.testing.assert_allclose(op+boundary_div,divergence,
                                   atol=3e-12,rtol=0)

    def test_ninepoint_matrix_symmetric_positive_definite_for_small_grid(self):
        a=assemble(7,45,"M1","sinesine").A.toarray()
        np.testing.assert_allclose(a,a.T,atol=0,rtol=0)
        self.assertGreater(float(np.linalg.eigvalsh(a)[0]),0.)


class IndependentCheckerSmokeTests(unittest.TestCase):
    def test_dst_solves_same_discrete_source_as_manual_stencil(self):
        n=8
        _,source=analytic_values(n)
        u,details=spectral_M0(n,source)
        self.assertLess(details["scaled_residual_linf"],1e-12)
        independent=independent_operator(u.ravel(),n,"M0")
        np.testing.assert_allclose(independent.reshape(n-1,n-1),source,
                                   atol=3e-12,rtol=0)

    def test_matrix_free_cg_matches_small_grid_exact_residual(self):
        n=8
        _,source=analytic_values(n)
        u,details=matrix_free_M1(n,source)
        self.assertLess(details["scaled_residual_linf"],1e-11)
        np.testing.assert_allclose(
            independent_operator(u.ravel(),n,"M1").reshape(n-1,n-1),source,
            atol=4e-11,rtol=0)

    def test_actual_small_linear_solve_matches_independent_matrix_free(self):
        n=8
        real=assemble(n,45,"M1","sinesine")
        solved,_=nodal_solution(real)
        _,rhs=analytic_values(n)
        other,report=matrix_free_M1(n,rhs)
        np.testing.assert_allclose(solved[1:-1,1:-1],other,atol=2e-11,rtol=0)
        self.assertLess(report["scaled_residual_linf"],1e-11)


if __name__ == "__main__":
    unittest.main()
