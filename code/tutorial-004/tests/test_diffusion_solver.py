"""Unit smoke checks for the tutorial-004 diffusion core; not formal ET004 runs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_solver import advance_one_step, _face_coefficients


class TestDiffusionCore(unittest.TestCase):
    def test_two_by_two_heat_hand_calculation(self):
        U = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(
            advance_one_step(U, dt=0.2),
            np.array([[0.6, 0.2], [0.2, 0.0]]), atol=1e-15
        )
        np.testing.assert_allclose(U, [[1, 0], [0, 0]], atol=0)

    def test_constant_image_fixed(self):
        U = np.full((12, 16), 0.42)
        for method in ("heat", "regularized"):
            np.testing.assert_allclose(
                advance_one_step(U, dt=0.2, method=method), U,
                rtol=0, atol=1e-15
            )

    def test_mass_range_variance_for_each_step(self):
        rng = np.random.Generator(np.random.PCG64(20261008))
        U0 = rng.normal(loc=0.5, scale=0.08, size=(32, 28))
        for method in ("heat", "regularized"):
            U = U0.copy()
            old_variance = np.sum((U - U.mean()) ** 2)
            for _ in range(45):
                U = advance_one_step(U, dt=0.2, method=method)
                self.assertLessEqual(abs(float(U.mean() - U0.mean())), 1e-12)
                self.assertGreaterEqual(float(U.min()), float(U0.min()) - 1e-12)
                self.assertLessEqual(float(U.max()), float(U0.max()) + 1e-12)
                variance = np.sum((U - U.mean()) ** 2)
                self.assertLessEqual(float(variance), float(old_variance) + 1e-9)
                old_variance = variance

    def test_linear_neumann_eigenmode(self):
        N = 64
        x = np.arange(N) + 0.5
        U = np.tile(0.5 + 0.1 * np.cos(np.pi * x / N), (N, 1))
        lam = -4.0 * np.sin(np.pi / (2 * N)) ** 2
        exact0 = U - 0.5
        errors = []
        for dt in (0.2, 0.1, 0.05, 0.025):
            W = U.copy()
            for _ in range(round(8 / dt)):
                W = advance_one_step(W, dt=dt, method="heat")
            expected = 0.5 + np.exp(lam * 8) * exact0
            errors.append(float(np.sqrt(np.mean((W - expected) ** 2))))
        orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
        self.assertTrue(all(0.85 < p < 1.15 for p in orders[-2:]))

    def test_invalid_time_step(self):
        U = np.ones((4, 4))
        for dt in (-1, 0, 0.26, float("nan")):
            with self.assertRaises(ValueError):
                advance_one_step(U, dt=dt)
        with self.assertRaises(ValueError):
            advance_one_step(U, dt=0.2, method="regularized", kappa=0)
        with self.assertRaises(ValueError):
            advance_one_step(U * float("nan"), dt=0.2)

    def test_face_weights_shapes_and_shared_values(self):
        U = np.arange(15, dtype=float).reshape(3, 5) / 30.0
        for method in ("heat", "regularized"):
            ax, ay = _face_coefficients(U, method=method, sigma=1.5, kappa=0.075, h=1.0)
            self.assertEqual(ax.shape, (3, 4))
            self.assertEqual(ay.shape, (2, 5))
            self.assertTrue(np.all((ax > 0) & (ax <= 1)))
            self.assertTrue(np.all((ay > 0) & (ay <= 1)))


if __name__ == "__main__":
    unittest.main()
