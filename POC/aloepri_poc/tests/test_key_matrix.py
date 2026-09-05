import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen

# (seed, d, h) cases: the original case, a couple more seeds on it, and two
# "tight bottleneck" shapes where h/2 is close to (or equal to) d, which is
# where a null-space construction is most likely to run out of room.
# h=0 is the POC-wide simplification (see plan) that Task 3 depends on;
# it used to crash with ZeroDivisionError (F1/F2's scale = sqrt(1/h)) before
# that was fixed alongside Task 3 discovering the dependency was broken.
INVARIANT_CASES = [
    (0, 16, 128),
    (7, 16, 128),
    (3, 32, 64),   # h/2 == d
    (5, 16, 32),   # h/2 == d
    (0, 16, 0),    # h=0: E/F/C/D degenerate to width 0, P_hat/Q_hat -> (d, d)
]


@pytest.mark.parametrize("seed,d,h", INVARIANT_CASES)
def test_key_matrices_are_exact_inverses(seed, d, h):
    rng = np.random.default_rng(seed)
    lam = 0.3
    base = init_key_matrix(d, h, lam, rng)

    p_hat = key_mat_gen(base)
    q_hat = inv_key_mat_gen(base)

    # Algorithm 1 (arXiv 2603.01499v2, p.8): P_hat = [B C E] Z is d x (d+2h),
    # Q_hat = Z^T [B^-1; F; D] is (d+2h) x d. Their product is the d x d
    # identity; the factors themselves are not square.
    assert p_hat.shape == (d, d + 2 * h)
    assert q_hat.shape == (d + 2 * h, d)
    np.testing.assert_allclose(p_hat @ q_hat, np.eye(d), atol=1e-5)


def test_two_calls_produce_different_matrices():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    d, h, lam = 16, 128, 0.3
    p1 = key_mat_gen(init_key_matrix(d, h, lam, rng1))
    p2 = key_mat_gen(init_key_matrix(d, h, lam, rng2))
    assert not np.allclose(p1, p2)


def test_sampling_scale_matches_paper():
    """Regression guard for the variance constants in Algorithm 1's Init step.

    The P_hat @ Q_hat = I invariant only depends on C being orthogonal to F's
    columns and D being orthogonal to E's columns (by construction) - it is
    blind to the *scale* used to sample V, E1, E2, F1, F2, so a wrong
    variance (e.g. a missing sqrt, or 1/sqrt(d) confused with 1/d) would not
    be caught by the invariant test above. Page 8 of the PDF states
    E1, E2 ~ N(0, 1/d) and F1, F2 ~ N(0, 1/h) (variance parameterization).
    E = E1 @ E2 and F = F1 @ F2 are already exposed on KeyMatrixBase, so we
    check their entrywise variance against the value implied by those
    constants, aggregated over many independent draws for a stable estimate.
    """
    d, h, lam = 16, 128, 0.3
    e_entries = []
    f_entries = []
    for seed in range(20):
        rng = np.random.default_rng(9000 + seed)
        base = init_key_matrix(d, h, lam, rng)
        e_entries.append(base.E.ravel())
        f_entries.append(base.F.ravel())
    e_entries = np.concatenate(e_entries)
    f_entries = np.concatenate(f_entries)

    half_h = h // 2
    expected_var_e = half_h / d**2        # Var(E1)*Var(E2) summed over h/2 terms
    expected_var_f = half_h / h**2        # Var(F1)*Var(F2) summed over h/2 terms

    np.testing.assert_allclose(e_entries.var(), expected_var_e, rtol=0.2)
    np.testing.assert_allclose(f_entries.var(), expected_var_f, rtol=0.2)
