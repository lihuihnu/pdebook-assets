"""First-order single-face finite volumes, without clipping or hidden fallback."""
import numpy as np
from burgers_flux import numerical_flux


def sine_primitive(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x - np.cos(2.0 * np.pi * x) / (8.0 * np.pi)


def periodic_initial_means(n):
    if type(n) is not int or n < 3:
        raise ValueError("periodic n must be an integer >= 3")
    faces = np.linspace(0.0, 1.0, n + 1)
    dx = 1.0 / n
    return (sine_primitive(faces[1:]) - sine_primitive(faces[:-1])) / dx


def riemann_initial_means(n, left, right):
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError("Riemann mesh n must be even")
    x = np.linspace(-1.0, 1.0, n + 1)
    dx = 2.0 / n
    length_left = np.clip(np.minimum(x[1:], 0.0) - x[:-1], 0.0, dx)
    return (left * length_left + right * (dx - length_left)) / dx


def face_fluxes(u, boundary, method="Godunov", left=None, right=None):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or u.size < 2 or not np.isfinite(u).all():
        raise ValueError("invalid one-dimensional state")
    face = np.empty(u.size + 1, dtype=np.float64)
    face[1:-1] = numerical_flux(u[:-1], u[1:], method)
    if boundary == "periodic":
        boundary_face = numerical_flux(u[-1], u[0], method)
        face[0] = boundary_face
        face[-1] = boundary_face  # same physical face, never recomputed
    elif boundary == "riemann":
        if left is None or right is None:
            raise ValueError("missing exterior Riemann states")
        face[0] = numerical_flux(left, u[0], method)
        face[-1] = numerical_flux(u[-1], right, method)
    else:
        raise ValueError("unknown boundary")
    return face


def euler_step(u, dx, dt, boundary, method="Godunov", left=None, right=None):
    if not (np.isfinite(dx) and dx > 0 and np.isfinite(dt) and dt > 0):
        raise ValueError("positive finite dx/dt required")
    cfl = dt / dx
    if cfl > 1.0 + 5e-14:
        raise ValueError("CFL > 1 rejected")
    old = np.asarray(u, dtype=np.float64)
    if not np.isfinite(old).all() or np.max(np.abs(old)) > 1.0 + 1e-12:
        raise ValueError("unregistered wave-speed bound")
    faces = face_fluxes(old, boundary, method, left, right)
    new = old - cfl * (faces[1:] - faces[:-1])
    if not np.isfinite(new).all():
        raise ValueError("nonfinite new state")
    lo, hi = float(np.min(old)), float(np.max(old))
    if boundary == "riemann":
        lo = min(lo, left, right)
        hi = max(hi, left, right)
    if (new.min() < lo - 1e-12) or (new.max() > hi + 1e-12):
        raise ValueError("discrete maximum-principle violation")
    return new, faces


def total_variation(u, boundary, left=None, right=None):
    u = np.asarray(u, dtype=np.float64)
    value = float(np.abs(np.diff(u)).sum())
    if boundary == "periodic":
        return value + abs(float(u[0] - u[-1]))
    if boundary == "riemann":
        return value + abs(float(u[0] - left)) + abs(float(right - u[-1]))
    raise ValueError("unknown boundary")
