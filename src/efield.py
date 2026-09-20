"""
efield.py -- point-charge electric fields and plotting helpers.

Implements the definitions from Halliday, Resnick & Krane, *Physics* Vol. 2, Ch. 26:

    E = F / q0                          (Eq. 26-3)
    F = q E                             (Eq. 26-4)
    E = (1/4*pi*eps0) |q| / r^2         (Eq. 26-6)
    E = sum_n E_n                       (Eq. 26-7, superposition)

All quantities are SI: coulombs, metres, newtons, N/C.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "EPS0", "K", "E_CHARGE", "M_ELECTRON", "M_PROTON", "A0",
    "coulomb_force", "E_point", "E_total", "force_on",
    "field_map", "field_lines", "trajectory",
]

# --------------------------------------------------------------------------
# Constants (CODATA 2018)
# --------------------------------------------------------------------------
EPS0 = 8.8541878128e-12           # C^2 / (N m^2)
K = 1.0 / (4 * np.pi * EPS0)      # 8.9875e9 N m^2 / C^2
E_CHARGE = 1.602176634e-19        # C
M_ELECTRON = 9.1093837015e-31     # kg
M_PROTON = 1.67262192369e-27      # kg
A0 = 5.29177210903e-11            # Bohr radius, m


# --------------------------------------------------------------------------
# Fields and forces
# --------------------------------------------------------------------------
def coulomb_force(q_src, r_src, q_test, r_test):
    """Force on ``q_test`` at ``r_test`` due to ``q_src`` at ``r_src``.

    Returns a 2-component numpy array in newtons.
    """
    d = np.asarray(r_test, float) - np.asarray(r_src, float)
    r = np.linalg.norm(d)
    return K * q_src * q_test * d / r ** 3


def E_point(q, r_q, X, Y, soft=1e-3):
    """Field of a single point charge ``q`` at ``r_q``, evaluated on ``X``, ``Y``.

    ``soft`` is a purely numerical softening radius: inside it the field is
    frozen so that a grid point landing on the charge does not divide by zero.
    It carries no physics -- keep it far below any length scale of interest,
    and set it to ~1e-9 for scalar (non-grid) evaluations.

    Returns ``(Ex, Ey)`` with the same shape as ``X``.
    """
    dx = np.asarray(X, float) - r_q[0]
    dy = np.asarray(Y, float) - r_q[1]
    r2 = np.maximum(dx ** 2 + dy ** 2, soft ** 2)
    r3 = r2 * np.sqrt(r2)
    return K * q * dx / r3, K * q * dy / r3


def E_total(charges, X, Y, soft=1e-3):
    """Superposed field of ``charges`` -- an iterable of ``(q, (x, y))`` pairs.

    Each charge's field is computed as if it were the only one present, then
    added vectorially (Eq. 26-7).
    """
    Ex = np.zeros_like(np.asarray(X, float))
    Ey = np.zeros_like(np.asarray(Y, float))
    for q, r_q in charges:
        ex, ey = E_point(q, r_q, X, Y, soft)
        Ex = Ex + ex
        Ey = Ey + ey
    return Ex, Ey


def force_on(q_test, r_test, charges, soft=1e-9):
    """Force on a test charge placed at ``r_test`` among ``charges`` (Eq. 26-4)."""
    Ex, Ey = E_total(charges, np.array(r_test[0]), np.array(r_test[1]), soft)
    return q_test * np.array([float(Ex), float(Ey)])


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _draw_charges(ax, charges):
    for q, (x, y) in charges:
        ax.plot(x, y, "o", ms=13, color="tab:red" if q > 0 else "tab:blue", zorder=5)
        ax.text(x, y, "+" if q > 0 else "\u2212", color="white", ha="center",
                va="center", fontweight="bold", zorder=6)


def field_map(charges, extent=0.10, n=26, ax=None, title="", soft=0.006):
    """Quiver plot: unit arrows show direction, colour shows log10(E)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))
    g = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(g, g)
    Ex, Ey = E_total(charges, X, Y, soft)

    mag = np.hypot(Ex, Ey)
    quiv = ax.quiver(X, Y, Ex / mag, Ey / mag, np.log10(mag), cmap="viridis",
                     pivot="mid", scale=32, width=0.004)
    plt.colorbar(quiv, ax=ax, shrink=0.8, label=r"$\log_{10} E$  (N/C)")

    _draw_charges(ax, charges)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), title=title,
           xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    return ax


def field_lines(charges, extent=0.10, n=300, ax=None, title="", density=1.6,
                soft=0.004):
    """Streamplot of the field lines -- curves everywhere tangent to E."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.4))
    g = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(g, g)
    Ex, Ey = E_total(charges, X, Y, soft)

    ax.streamplot(X, Y, Ex, Ey, color=np.log10(np.hypot(Ex, Ey)), cmap="plasma",
                  density=density, linewidth=0.9, arrowsize=0.9)
    _draw_charges(ax, charges)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), title=title,
           xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    return ax


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------
def trajectory(charges, q, m, r0, v0, t_end, n_pts=2000, soft=2e-3):
    """Integrate the motion of a charge ``q`` of mass ``m`` through the field.

    Uses F = qE with the source charges held fixed; the moving charge's own
    field is ignored. Returns the ``scipy`` solution object, with positions in
    ``sol.y[0]``, ``sol.y[1]``.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, s):
        x, y, vx, vy = s
        Ex, Ey = E_total(charges, np.array(x), np.array(y), soft)
        return [vx, vy, q * float(Ex) / m, q * float(Ey) / m]

    return solve_ivp(rhs, (0, t_end), [*r0, *v0],
                     t_eval=np.linspace(0, t_end, n_pts),
                     rtol=1e-8, atol=1e-10, method="DOP853")
