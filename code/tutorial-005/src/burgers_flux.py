"""Convex Burgers fluxes; Godunov is the primary conservative method."""
import numpy as np


def physical(u):
    v = np.asarray(u, dtype=np.float64)
    return 0.5 * v * v


def numerical_flux(a, b, method="Godunov"):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite face state")
    if method == "Godunov":
        # Convex flux: minimum for rarefaction, maximum for shock.
        out = np.where(a <= b,
                       np.where(a >= 0.0, physical(a),
                                np.where(b <= 0.0, physical(b), 0.0)),
                       np.maximum(physical(a), physical(b)))
    elif method == "Roe_no_entropy_fix":
        out = 0.5 * (physical(a) + physical(b)) - 0.5 * np.abs(0.5 * (a + b)) * (b - a)
    elif method == "LF_alpha1":
        out = 0.5 * (physical(a) + physical(b)) - 0.5 * (b - a)
    else:
        raise ValueError("unregistered flux: " + str(method))
    return out
