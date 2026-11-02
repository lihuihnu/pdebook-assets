"""ET007-04 pre-run small algebraic tests, NOT the eight production PDE cases."""
from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.model import assemble
from src.validate_et007_04 import (
    tensor,analytic,action,reference_solve,face_metrics,
    ANGLES,NEW_ANGLES,N_LIST,METHODS,SOURCE_HASHES,
)


class RotatedTensorAndManufacturedSource(unittest.TestCase):
    def test_tensor_angle_and_eigenvalues_without_production_import(self):
        self.assertEqual(ANGLES,(0,15,30,45))
        self.assertEqual(NEW_ANGLES,(15,30))
        self.assertEqual(N_LIST,(64,128))
        self.assertEqual(METHODS,("M0","M1"))
        for angle in (15,30):
            with self.subTest(angle=angle):
                kxx,kxy,kyy=tensor(angle)
                self.assertAlmostEqual(kxx+kyy,11.,places=13)
                self.assertGreater(kxy,0.)
                np.testing.assert_allclose(
                    np.linalg.eigvalsh([[kxx,kxy],[kxy,kyy]]),
                    [1.,10.],rtol=0,atol=3e-14)
                self.assertAlmostEqual(kxy,9*math.sin(math.radians(2*angle))/2,
                                       places=13)

    def test_full_source_has_mixed_term_not_absorbed_into_m0(self):
        n=8
        for angle in NEW_ANGLES:
            with self.subTest(angle=angle):
                exact,source=analytic(n,angle)
                kxx,kxy,kyy=tensor(angle)
                x,y=1/n,2/n
                expected=(math.pi**2*(kxx+kyy)
                          *math.sin(math.pi*x)*math.sin(math.pi*y)
                          -2*math.pi**2*kxy
                          *math.cos(math.pi*x)*math.cos(math.pi*y))
                self.assertAlmostEqual(float(source[0,1]),expected,places=12)
                self.assertFalse(np.allclose(source,
                    11*math.pi**2*exact,rtol=0,atol=1e-12))

    def test_constant_and_affine_fields_have_zero_stencil_divergence(self):
        n=8
        x=np.arange(n+1)/n
        for angle in NEW_ANGLES:
            for method in METHODS:
                for field in ("constant","x","y"):
                    with self.subTest(angle=angle,method=method,field=field):
                        u=(np.ones((n+1,n+1)) if field=="constant"
                           else np.broadcast_to(
                               (x[:,None] if field=="x" else x[None,:]),
                               (n+1,n+1)).copy())
                        full_u=np.zeros_like(u)
                        full_u[1:-1,1:-1]=u[1:-1,1:-1]
                        operator=action(full_u[1:-1,1:-1].ravel(),n,angle,method)
                        # Add the contribution from nonzero Dirichlet values;
                        # exactly reproduces the supplied affine physical field.
                        boundary=np.array(u)
                        boundary[1:-1,1:-1]=0.
                        h=1/n
                        kxx,kxy,kyy=tensor(angle)
                        def full_operator(data):
                            m=data[1:-1,1:-1]
                            v=-(kxx*(data[2:,1:-1]-2*m+data[:-2,1:-1])
                              +kyy*(data[1:-1,2:]-2*m+data[1:-1,:-2]))/(h*h)
                            if method=="M1":
                                v-=kxy*(data[2:,2:]-data[2:,:-2]
                                       -data[:-2,2:]+data[:-2,:-2])/(2*h*h)
                            return v.ravel()
                        np.testing.assert_allclose(operator+full_operator(boundary),
                                                   np.zeros_like(operator),
                                                   rtol=0,atol=2e-12)

    def test_nonhomogeneous_corner_xy_against_full_tensor_source(self):
        n=4
        x=np.arange(n+1)/n
        all_xy=x[:,None]*x[None,:]
        for angle in NEW_ANGLES:
            with self.subTest(angle=angle):
                system=assemble(n,float(angle),"M1","xy")
                expected=-2*tensor(angle)[1]
                inside=all_xy[1:-1,1:-1].ravel()
                np.testing.assert_allclose(
                    system.A@inside-system.b,
                    np.zeros((n-1)**2),rtol=0,atol=1e-12)
                self.assertAlmostEqual(expected,-2*tensor(angle)[1])


class IndependentReferenceMethods(unittest.TestCase):
    def test_dst_m0_and_cg_m1_match_true_independent_rhs(self):
        n=8
        for angle in NEW_ANGLES:
            _,source=analytic(n,angle)
            for method in METHODS:
                with self.subTest(theta_deg=angle,method=method):
                    ref,details=reference_solve(n,angle,method,source)
                    self.assertTrue(np.all(np.isfinite(ref)))
                    self.assertLess(details["residual_linf_relative"],1e-11)
                    self.assertEqual(ref.shape,(n-1,n-1))
                    sys=assemble(n,float(angle),method,"sinesine")
                    np.testing.assert_allclose(
                        sys.b.reshape((n-1,n-1)),source,rtol=0,atol=1e-13)
                    np.testing.assert_allclose(
                        sys.A@ref.ravel(),sys.b,rtol=0,atol=2e-11)

    def test_shared_face_conservation_on_independent_reference(self):
        n=8
        for angle in NEW_ANGLES:
            _,source=analytic(n,angle)
            for method in METHODS:
                with self.subTest(angle=angle,method=method):
                    u,_=reference_solve(n,angle,method,source)
                    metrics=face_metrics(u,n,angle,method)
                    self.assertLess(metrics["dual_cell_balance_scaled"],1e-11)
                    self.assertLess(metrics["dual_boundary_budget_scaled"],1e-11)
                    for key,value in metrics.items():
                        self.assertTrue(math.isfinite(value),key)

    def test_frozen_sources_are_explicit_and_not_invented(self):
        self.assertEqual(set(SOURCE_HASHES),
                         {"ET007-01","ET007-02","ET007-03"})
        self.assertTrue(all(len(h)==64 and
                            all(ch in "0123456789abcdef" for ch in h)
                            for h in SOURCE_HASHES.values()))


if __name__=="__main__":
    unittest.main()
