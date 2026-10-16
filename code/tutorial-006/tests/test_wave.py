"""Independent identities and boundary/initialization checks for wave.py."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wave import (discrete_inner, displacement_l2, initial_step, integer_step,
                      leapfrog_step, pulse, pulse_derivative, solve_wave)


class PulseTests(unittest.TestCase):
    def test_support_symmetry_and_peak(self):
        x = np.array([0.1, 0.25, 0.30, 0.35, 0.40, 0.45, 0.8])
        values = pulse(x)
        np.testing.assert_allclose(values[[0, 1, 5, 6]], np.zeros(4),
                                   rtol=0.0, atol=1e-70)
        self.assertEqual(values[3], 1.0)
        self.assertAlmostEqual(values[2], values[4], places=14)
        self.assertAlmostEqual(values[2], 0.75 ** 6, places=14)

    def test_analytic_derivative_against_centered_difference(self):
        x = np.array([0.27, 0.31, 0.35, 0.38, 0.43])
        epsilon = 1e-6
        numerical = (pulse(x + epsilon) - pulse(x - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(pulse_derivative(x), numerical,
                                   atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(pulse_derivative(np.array([0.1, 0.25, 0.45, 0.8])),
                                   np.zeros(4), rtol=0.0, atol=1e-60)

    def test_right_moving_velocity_sign(self):
        x = np.array([0.30, 0.40])
        velocity = -pulse_derivative(x)
        self.assertLess(velocity[0], 0.0)
        self.assertGreater(velocity[1], 0.0)


class StencilTests(unittest.TestCase):
    def test_taylor_start_exact_for_quadratic_displacement_linear_velocity(self):
        # u=x^2+t*(0.2+0.3*x)+t^2 solves u_tt=u_xx before boundary effects.
        x = np.linspace(-1.0, 1.0, 9)
        dt = 0.1
        h = 0.25
        velocity = 0.2 + 0.3 * x
        actual = initial_step(x * x, velocity, dt / h, dt)
        exact = x * x + dt * velocity + dt * dt
        np.testing.assert_allclose(actual[1:-1], exact[1:-1],
                                   rtol=0.0, atol=4e-16)

    def test_smallest_grid_and_new_interior_endpoint_dependency(self):
        previous = np.array([0.0, 1.0, 0.0])
        current = np.array([0.0, 0.5, 0.0])
        fixed = leapfrog_step(previous, current, 0.5)
        np.testing.assert_allclose(fixed, [0.0, -0.25, 0.0],
                                   rtol=0.0, atol=0.0)
        # q=-1/3; the freshly computed -1/4 interior yields 1/2+1/12.
        mur = leapfrog_step(previous, current, 0.5, "mur", 0.5)
        np.testing.assert_allclose(mur, [7 / 12, -0.25, 7 / 12],
                                   rtol=0.0, atol=2e-16)

    def test_left_and_right_boundaries_are_mirror_images(self):
        previous = np.array([0.0, 0.2, -0.1, 0.5, 0.3, 0.0])
        current = np.array([0.0, -0.3, 0.4, 0.2, 0.1, 0.0])
        for boundary, cb in (("dirichlet", None), ("mur", 0.4),
                             ("mur", 0.8), ("mur", 1.6)):
            with self.subTest(boundary=boundary, cb=cb):
                right = leapfrog_step(previous, current, 0.8, boundary, cb)
                left = leapfrog_step(previous[::-1], current[::-1], 0.8, boundary, cb)
                np.testing.assert_allclose(right, left[::-1], rtol=0.0, atol=3e-16)
                started = initial_step(previous, current, 0.8, 0.1, boundary, cb)
                mirrored = initial_step(previous[::-1], current[::-1],
                                        0.8, 0.1, boundary, cb)
                np.testing.assert_allclose(started, mirrored[::-1], rtol=0.0, atol=3e-16)

    def test_courant_one_exact_shift_with_independent_two_layer_initialization(self):
        # Deliberately do NOT use the Taylor start in this exact shift identity.
        # An arbitrary binary-exact sequence is independently shifted once.
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                previous = np.zeros(64)
                previous[20:24] = [0.25, 1.0, -0.5, 0.125]
                def shift(values):
                    if direction == 1:
                        return np.concatenate(([0.0], values[:-1]))
                    return np.concatenate((values[1:], [0.0]))
                current = shift(previous)
                for _ in range(80):
                    expected = shift(current)
                    new = leapfrog_step(previous, current, 1.0, "mur", 1.0)
                    np.testing.assert_array_equal(new, expected)
                    previous, current = current, new

    def test_zero_solution_is_preserved(self):
        zero = np.zeros(11)
        for boundary, cb in (("dirichlet", None), ("mur", 0.8)):
            np.testing.assert_array_equal(
                initial_step(zero, zero, 0.8, 0.01, boundary, cb), zero)
            np.testing.assert_array_equal(
                leapfrog_step(zero, zero, 0.8, boundary, cb), zero)

    def test_invalid_courant_is_rejected_including_above_one(self):
        zero = np.zeros(5)
        for courant in (0.0, -0.2, 1.00001, float("nan"), float("inf")):
            with self.subTest(courant=courant):
                with self.assertRaises(ValueError):
                    leapfrog_step(zero, zero, courant)
                with self.assertRaises(ValueError):
                    initial_step(zero, zero, courant, 0.1)
                with self.assertRaises(ValueError):
                    solve_wave(200, courant=courant)

    def test_invalid_boundary_and_state_are_rejected(self):
        zero = np.zeros(5)
        for cb in (None, 0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                leapfrog_step(zero, zero, 0.8, "mur", cb)
        with self.assertRaises(ValueError):
            leapfrog_step(zero, zero, 0.8, "unknown")
        with self.assertRaises(ValueError):
            leapfrog_step(zero, np.zeros(4), 0.8)
        with self.assertRaises(ValueError):
            leapfrog_step(np.zeros(2), np.zeros(2), 0.8)


class GridAndDiagnosticTests(unittest.TestCase):
    def test_composite_trapezoid_constant_and_endpoint_weights(self):
        self.assertAlmostEqual(displacement_l2(np.ones(5), 0.25), 1.0)
        left_endpoint = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(discrete_inner(left_endpoint, left_endpoint, 0.25), 0.125)
        a = np.array([1.0, -2.0, 3.0])
        b = np.array([-1.0, 4.0, 2.0])
        self.assertAlmostEqual(discrete_inner(a, b, 0.5), -2.75)

    def test_fixed_step_times_reject_clipping(self):
        for n in (200, 400, 800, 1600):
            dt = 0.8 / n
            self.assertEqual(integer_step(1.2, dt), 3 * n // 2)
            for time in (0.0, 0.4, 0.6, 0.9, 1.2):
                self.assertAlmostEqual(integer_step(time, dt) * dt, time)
        with self.assertRaises(ValueError):
            integer_step(0.123, 0.004)
        with self.assertRaises(ValueError):
            solve_wave(200, final_time=0.123, snapshot_times=(0.0,))

    def test_large_crop_uses_exact_short_grid_and_fixed_dt(self):
        options = dict(final_time=0.04, snapshot_times=(0.0, 0.04))
        short = solve_wave(200, domain=(0, 1), **options)
        large = solve_wave(200, domain=(-2, 3), **options)
        np.testing.assert_array_equal(short.x, large.x)
        self.assertEqual(large.crop_start, 400)
        self.assertEqual(large.full_nodes, 1001)
        self.assertEqual(short.dt, large.dt)
        self.assertEqual(len(short.diagnostics), short.final_step + 1)
        for step in short.snapshots:
            np.testing.assert_array_equal(short.snapshots[step], large.snapshots[step])

    def test_initial_pulse_and_requested_snapshots_are_saved(self):
        result = solve_wave(200, final_time=0.04, snapshot_times=(0.0, 0.02, 0.04),
                            boundary="mur", boundary_speed_ratio=1.0)
        self.assertEqual(set(result.snapshots), {0, 5, 10})
        np.testing.assert_array_equal(result.snapshots[0], pulse(result.x))
        self.assertTrue(all(math.isfinite(row[-1]) for row in result.diagnostics))
        self.assertEqual(result.diagnostics[-1][0], 10)

    def test_invalid_grid_crop_and_snapshot_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_wave(1)
        with self.assertRaises(ValueError):
            solve_wave(200, crop=(-1, 1))
        with self.assertRaises(ValueError):
            solve_wave(200, snapshot_times=(0.0, 1.6))
        with self.assertRaises(ValueError):
            solve_wave(200, snapshot_times=(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
