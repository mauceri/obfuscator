import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from block_perm import block_perm


def test_block_perm_is_a_valid_permutation_matrix():
    m_blocks = 12
    z = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=m_blocks, seed=0)
    assert z.shape == (m_blocks, m_blocks)
    # chaque ligne et chaque colonne a exactement un 1, le reste des 0
    row_sums = z.sum(dim=1)
    col_sums = z.sum(dim=0)
    torch.testing.assert_close(row_sums, torch.ones(m_blocks))
    torch.testing.assert_close(col_sums, torch.ones(m_blocks))
    assert set(z.unique().tolist()) <= {0.0, 1.0}


def test_block_perm_respects_max_window_size():
    # Avec une fenêtre max de 1, chaque bloc doit rester à sa place
    # (aucune permutation possible au-delà d'un singleton).
    m_blocks = 10
    z = block_perm(beta=1, gamma=0.5, zeta=1e3, m_blocks=m_blocks, seed=1)
    torch.testing.assert_close(z, torch.eye(m_blocks))


def test_block_perm_is_reproducible_with_same_seed():
    z1 = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=12, seed=42)
    z2 = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=12, seed=42)
    torch.testing.assert_close(z1, z2)
