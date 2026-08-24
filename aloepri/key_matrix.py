"""Algorithm 1 (Key Matrix Generation) from AloePri (arXiv 2603.01499v2, p.8).

Note: P̂ and Q̂ have shapes (d, d+2h) and (d+2h, d) respectively, not (d, d)
as initially transcribed. Shapes derived from dimensional analysis of the
algorithm and verified against the primary source.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


@dataclass
class KeyMatrixBase:
    B: np.ndarray
    B_inv: np.ndarray
    E: np.ndarray
    F: np.ndarray
    Z: np.ndarray
    rng: np.random.Generator


def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform sample from the orthogonal group O_n (QR + sign fix)."""
    g = rng.normal(size=(n, n))
    q, r = np.linalg.qr(g)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return q * sign


def _null_space_samples(matrix: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` random vectors drawn from null(matrix), as columns of the result."""
    basis = null_space(matrix)
    coeffs = rng.normal(size=(basis.shape[1], count))
    return basis @ coeffs


def init_key_matrix(d: int, h: int, lam: float, rng: np.random.Generator) -> KeyMatrixBase:
    half_h = h // 2

    U = _random_orthogonal(d, rng)
    V = rng.normal(scale=np.sqrt(1.0 / d), size=(d, d))
    B = U + lam * V
    B_inv = np.linalg.inv(B)

    E1 = rng.normal(scale=np.sqrt(1.0 / d), size=(d, half_h))
    E2 = rng.normal(scale=np.sqrt(1.0 / d), size=(half_h, h))
    E = E1 @ E2

    # h=0 (POC-wide simplification, see plan) collapses F1/F2 to empty
    # arrays; guard the scale division so it doesn't ZeroDivisionError on a
    # value that no sampled element will ever use.
    f_scale = np.sqrt(1.0 / h) if h > 0 else 0.0
    F1 = rng.normal(scale=f_scale, size=(h, half_h))
    F2 = rng.normal(scale=f_scale, size=(half_h, d))
    F = F1 @ F2

    Z = _random_orthogonal(d + 2 * h, rng)

    return KeyMatrixBase(B=B, B_inv=B_inv, E=E, F=F, Z=Z, rng=rng)


def key_mat_gen(base: KeyMatrixBase) -> np.ndarray:
    """P_hat = [B C E] Z, with C's rows sampled from null(F^T) so that C @ F = 0."""
    d = base.B.shape[0]
    C = _null_space_samples(base.F.T, d, base.rng).T
    return np.hstack([base.B, C, base.E]) @ base.Z


def inv_key_mat_gen(base: KeyMatrixBase) -> np.ndarray:
    """Q_hat = Z^T [B^-1; F; D], with D's columns sampled from null(E) so that E @ D = 0."""
    d = base.B.shape[0]
    D = _null_space_samples(base.E, d, base.rng)
    return base.Z.T @ np.vstack([base.B_inv, base.F, D])
