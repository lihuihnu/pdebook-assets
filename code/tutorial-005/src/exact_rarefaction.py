"""Exact entropy rarefaction reference: integral of -1, x/t, 1, not point samples."""
import numpy as np


def entropy_rarefaction_cell_means(n, t):
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError("even positive Riemann grid required")
    if not np.isfinite(t) or not (0.0 <= t <= 0.5):
        raise ValueError("reference restricted to registered t in [0,0.5]")
    faces = np.linspace(-1.0, 1.0, n + 1, dtype=np.float64)
    if t == 0.0:
        # The interface is at x=0, exactly a grid face for even n.
        return np.where(faces[:-1] < 0.0, -1.0, 1.0)
    antiderivative = np.where(
        faces < -t, -faces - 0.5 * t,
        np.where(faces <= t, faces * faces / (2.0 * t), faces - 0.5 * t))
    return (antiderivative[1:] - antiderivative[:-1]) / (2.0 / n)
