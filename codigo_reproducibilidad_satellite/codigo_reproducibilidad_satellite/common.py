"""Funciones comunes para reproducir los análisis del sistema de actitud caótica.

Sistema (Ec. 9):
    wx_dot = wz*wy/3 - a*wx + wz/sqrt(6)
    wy_dot = -wz*wx + b*wy
    wz_dot = wx*wy - sqrt(6)*wx - c*wz

Sección de Poincaré usada en todos los scripts:
    wz = 0,  wz_dot > 0
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

SQRT6 = np.sqrt(6.0)
DEFAULT_IC = np.array([0.15, 0.15, 0.15], dtype=float)


def rhs(t, y, a=0.4, b=0.175, c=0.4):
    wx, wy, wz = y
    return np.array([
        (wz * wy) / 3.0 - a * wx + wz / SQRT6,
        -wz * wx + b * wy,
        wx * wy - SQRT6 * wx - c * wz,
    ], dtype=float)


def jacobian(y, a=0.4, b=0.175, c=0.4):
    wx, wy, wz = y
    return np.array([
        [-a, wz/3.0, wy/3.0 + 1.0/SQRT6],
        [-wz, b, -wx],
        [wy-SQRT6, wx, -c],
    ], dtype=float)


def poincare_sequence(a=0.4, b=0.175, c=0.4, *,
                      tmax=1200.0, transient=400.0, y0=DEFAULT_IC,
                      rtol=2e-9, atol=2e-11, max_step=0.12):
    """Integra el flujo y devuelve cruces ascendentes con wz=0.

    Returns
    -------
    times : (N,) ndarray
    points : (N,3) ndarray
    """
    def event(t, y):
        return y[2]
    event.direction = 1
    event.terminal = False

    sol = solve_ivp(lambda t, y: rhs(t, y, a, b, c),
                    (0.0, tmax), np.asarray(y0, float),
                    method="DOP853", rtol=rtol, atol=atol,
                    max_step=max_step, events=event)
    te = sol.t_events[0]
    ye = sol.y_events[0]
    mask = te >= transient
    return te[mask], ye[mask]


def _advance_off_section(q, a, b, c, dt=0.05,
                         rtol=2e-10, atol=2e-12):
    """Avanza brevemente desde z=0 para evitar contar el cruce t=0."""
    y0 = np.array([q[0], q[1], 0.0], dtype=float)
    sol = solve_ivp(lambda t, y: rhs(t, y, a, b, c),
                    (0.0, dt), y0, method="DOP853",
                    rtol=rtol, atol=atol, max_step=dt/5)
    return sol.y[:, -1]


def poincare_map(q, a=0.4, b=0.175, c=0.4, *,
                 rtol=2e-10, atol=2e-12, max_step=0.08, max_time=40.0):
    """Mapa de Poincaré 2D: q=(wx,wy) -> siguiente cruce ascendente wz=0."""
    ys = _advance_off_section(q, a, b, c, rtol=rtol, atol=atol)

    def event(t, y):
        return y[2]
    event.direction = 1
    event.terminal = True

    sol = solve_ivp(lambda t, y: rhs(t, y, a, b, c),
                    (0.05, max_time), ys, method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step, events=event)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("No se encontró el siguiente cruce de Poincaré.")
    return sol.y_events[0][0][:2]


def iterate_map(q, k, a=0.4, b=0.175, c=0.4):
    q = np.asarray(q, float)
    for _ in range(k):
        q = poincare_map(q, a=a, b=b, c=c)
    return q


def map_jacobian(q, k=1, a=0.4, b=0.175, c=0.4, h=2e-6):
    """Jacobiano numérico de P^k por diferencias centrales."""
    q = np.asarray(q, float)
    M = np.zeros((2, 2), dtype=float)
    for j in range(2):
        dq = np.zeros(2); dq[j] = h
        p_plus = iterate_map(q + dq, k, a=a, b=b, c=c)
        p_minus = iterate_map(q - dq, k, a=a, b=b, c=c)
        M[:, j] = (p_plus - p_minus) / (2*h)
    return M


def find_periodic_orbit(q_guess, k, a=0.4, b=0.175, c=0.4, tol=1e-10):
    """Resuelve P^k(q)-q=0 en la sección."""
    q_guess = np.asarray(q_guess, float)
    def F(q):
        return iterate_map(q, k, a=a, b=b, c=c) - q
    sol = root(F, q_guess, tol=tol)
    if not sol.success or np.linalg.norm(F(sol.x)) > 5e-7:
        raise RuntimeError(f"No convergió la órbita periódica k={k}: {sol.message}")
    return sol.x


def floquet_multipliers_map(q, k=1, a=0.4, b=0.175, c=0.4, h=2e-6):
    """Multiplicadores del mapa P^k (2D), equivalentes a los Floquet no triviales."""
    M = map_jacobian(q, k=k, a=a, b=b, c=c, h=h)
    return np.linalg.eigvals(M), M


def period_signature(x, ks=(1,2,4,8,16)):
    x = np.asarray(x, float)
    out = {}
    for k in ks:
        out[k] = np.sqrt(np.mean((x[k:] - x[:-k])**2)) if len(x) > k else np.nan
    return out


def classify_period(x, tol=1e-4):
    sig = period_signature(x)
    for k in (1,2,4,8,16):
        if np.isfinite(sig[k]) and sig[k] < tol:
            return k, sig
    return 0, sig


def robust_mad(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))
