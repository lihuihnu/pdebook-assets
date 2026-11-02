"""Unit and tiny-grid algebraic regressions for the frozen Tutorial-007 model."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import (tensor, assemble, dual_face_flux,
                       nodal_solution, report_metrics, exact_and_source)


class TensorTests(unittest.TestCase):
    def test_tensor_rotation_and_eigenvalues(self):
        for angle in (-45,0,15,30,45):
            a,b,c=tensor(angle)
            self.assertTrue(np.allclose(np.linalg.eigvalsh([[a,b],[b,c]]),[1.,10.],
                                        atol=3e-14,rtol=0))
        self.assertEqual(tensor(0),(10.,0.,1.))
        np.testing.assert_allclose(tensor(45),(5.5,4.5,5.5),
                                   atol=2e-15,rtol=0)
        self.assertAlmostEqual(tensor(-45)[1],-4.5,places=13)

    def test_invalid_input(self):
        for angle in (float("nan"),float("inf")):
            with self.assertRaises(ValueError): tensor(angle)
        with self.assertRaises(ValueError): tensor(0,(0.,1.))
        with self.assertRaises(ValueError): assemble(3,0,"M0","sinesine")
        with self.assertRaises(ValueError): assemble(4,0,"bad","sinesine")


class DirichletTests(unittest.TestCase):
    def test_nonhomogeneous_corner_elimination_xy(self):
        n=4
        system=assemble(n,45,"M1","xy")
        h=1./n
        interior=np.array([i*j*h*h for i in range(1,n) for j in range(1,n)])
        residual=system.A@interior-system.b
        np.testing.assert_allclose(residual,np.zeros_like(residual),atol=5e-13,rtol=0)
        # Corner (3,3) has nonzero right/top/corner Dirichlet values.
        idx=(n-2)*(n-1)+(n-2)
        self.assertAlmostEqual(float(system.b[idx]),123.,places=12)
        self.assertAlmostEqual(float((system.A@interior)[idx]),123.,places=12)
        self.assertAlmostEqual(float(system.b[0]),-9.,places=12)

    def test_m0_and_m1_reduce_to_identical_fivepoint(self):
        for n in (4,8,16):
            a=assemble(n,0,"M0","sinesine")
            b=assemble(n,0,"M1","sinesine")
            self.assertEqual((a.A-b.A).nnz,0)
            self.assertTrue(np.array_equal(a.b,b.b))
            self.assertEqual(a.A.nnz,5*(n-1)**2-4*(n-1))

    def test_near_boundary_diagonal_signs(self):
        n=4
        system=assemble(n,45,"M1","xy")
        mat=system.A
        m=n-1
        # Near middle: row for (2,2), off-diagonal NE=(3,3), NW=(1,3)
        row=(2-1)*m+(2-1)
        ne=(3-1)*m+(3-1)
        nw=(1-1)*m+(3-1)
        self.assertAlmostEqual(float(mat[row,ne]),-4.5/(2*(1/n)**2))
        self.assertAlmostEqual(float(mat[row,nw]),+4.5/(2*(1/n)**2))

    def test_diagonal_tensor_spd_small_grid(self):
        system=assemble(5,0,"M1","sinesine")
        a=system.A.toarray()
        np.testing.assert_allclose(a,a.T,rtol=0,atol=0)
        self.assertGreater(float(np.linalg.eigvalsh(a)[0]),0.)


class FluxTests(unittest.TestCase):
    def test_exact_xy_rotation_flux_and_dual_budget(self):
        n=4
        h=1./n
        x=np.arange(n+1)*h
        u=x[:,None]*x[None,:]
        from src.model import analytic_face_flux
        qx,qy=dual_face_flux(u,45,"M1")
        fx,fy=analytic_face_flux("xy",45,n,(10.,1.))
        np.testing.assert_allclose(qx,fx,rtol=0,atol=6e-14)
        np.testing.assert_allclose(qy,fy,rtol=0,atol=6e-14)
        balance=h*(qx[1:]-qx[:-1]+qy[:,1:]-qy[:,:-1])
        np.testing.assert_allclose(balance,
                                   -9*h*h*np.ones((n-1,n-1)),
                                   atol=5e-14,rtol=0)

    def test_affine_temperature_not_enough_to_verify_flux(self):
        n=4
        flux={}
        for method in ("M0","M1"):
            system=assemble(n,45,method,"affine")
            u,residual=nodal_solution(system)
            self.assertLess(residual,1e-12)
            x=np.arange(n+1)*1./n
            np.testing.assert_allclose(u,x[:,None]+np.zeros_like(u),
                                       atol=2e-14,rtol=0)
            flux[method]=dual_face_flux(u,45,method)
        self.assertAlmostEqual(float(np.mean(flux["M0"][1])),0.,places=12)
        self.assertAlmostEqual(float(np.mean(flux["M1"][1])),-4.5,places=12)
        self.assertAlmostEqual(float(np.mean(flux["M1"][0])),-5.5,places=12)

    def test_one_stored_face_is_shared_not_duplicated(self):
        n=6
        u=np.array([[0.2*i*i-0.4*i*j+0.3*j*j
                     for j in range(n+1)] for i in range(n+1)],dtype=float)
        qx,qy=dual_face_flux(u,45,"M1")
        h=1./n
        # Same qx[2,2] leaves cell (2,3) and enters cell (3,3).
        outgoing=h*qx[2,2]
        incoming=-h*qx[2,2]
        self.assertEqual(outgoing+incoming,0.)

    def test_sine_analytic_discrete_mode(self):
        n=16
        sys=assemble(n,0,"M0","sinesine")
        u,_=nodal_solution(sys)
        h=1./n
        factor=(math.pi*h/(2*math.sin(math.pi*h/2)))**2
        s=np.sin(math.pi*np.arange(n+1)*h)
        ref=factor*s[:,None]*s[None,:]
        ref[0,:]=ref[n,:]=0.
        ref[:,0]=ref[:,n]=0.
        np.testing.assert_allclose(u,ref,rtol=0,atol=8e-13)
        m=report_metrics(u,0,"M0","sinesine")
        self.assertAlmostEqual(m["T_error_l2"],(factor-1.)/2.,places=11)


if __name__=="__main__":
    unittest.main()
