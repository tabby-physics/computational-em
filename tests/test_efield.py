"""Physics checks for efield.py -- each test mirrors a claim made in HRK Ch. 26."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efield import (  # noqa: E402
    A0, E_CHARGE, K, E_point, E_total, coulomb_force, force_on,
)


def test_field_independent_of_test_charge():
    """Eq. 26-3: E = F/q0 must not depend on the size of q0."""
    q, rP = 2.0e-9, np.array([0.03, 0.04])
    values = []
    for q0 in (1e-12, 1e-10, 1e-9, 1e-8):
        F = coulomb_force(q, [0, 0], q0, rP)
        values.append(F / q0)
    for v in values[1:]:
        assert np.allclose(v, values[0])


def test_inverse_square_law():
    """Eq. 26-6: log-log slope of E(r) is -2."""
    r = np.linspace(0.01, 0.20, 200)
    Ex, Ey = E_point(1e-9, (0, 0), r, np.zeros_like(r))
    slope = np.polyfit(np.log(r), np.log(np.hypot(Ex, Ey)), 1)[0]
    assert slope == pytest.approx(-2.0, abs=1e-6)


def test_direction_reverses_with_sign():
    """E points outward from a positive charge, inward toward a negative one."""
    pos = np.array(E_point(+1e-9, (0, 0), 0.05, 0.0, soft=1e-9))
    neg = np.array(E_point(-1e-9, (0, 0), 0.05, 0.0, soft=1e-9))
    assert np.allclose(pos, -neg)
    assert pos[0] > 0 and neg[0] < 0


def test_newtons_third_law_with_unequal_charges():
    """Fig. 26-2: F12 = -F21 even though E1 != E2."""
    q1, rA = 3.0e-9, (0.0, 0.0)
    q2, rB = -1.0e-9, (0.05, 0.0)

    E1_at_B = np.array(E_point(q1, rA, rB[0], rB[1], soft=1e-9))
    E2_at_A = np.array(E_point(q2, rB, rA[0], rA[1], soft=1e-9))

    assert not np.isclose(np.linalg.norm(E1_at_B), np.linalg.norm(E2_at_A))
    assert np.allclose(q2 * E1_at_B, -(q1 * E2_at_A))


def test_superposition_matches_direct_coulomb_sum():
    """Eq. 26-7 against a term-by-term Coulomb sum."""
    charges = [(2e-9, (-0.03, 0.0)), (-4e-9, (0.03, 0.02))]
    rP, qP = (0.01, -0.02), 1e-9
    direct = sum(coulomb_force(q, rq, qP, rP) for q, rq in charges)
    assert np.allclose(force_on(qP, rP, charges), direct)


def test_null_point_between_like_charges():
    """Two equal like charges cancel exactly at their midpoint."""
    d = 0.03
    like = [(4e-9, (-d, 0.0)), (4e-9, (d, 0.0))]
    Ex, Ey = E_total(like, np.array(0.0), np.array(0.0), soft=1e-9)
    assert float(Ex) == pytest.approx(0.0, abs=1e-9)
    assert float(Ey) == pytest.approx(0.0, abs=1e-9)


def test_hydrogen_field_matches_table_26_1():
    """Table 26-1 quotes 5e11 N/C at the electron's average radius."""
    E = K * E_CHARGE / A0 ** 2
    assert 4e11 < E < 6e11
