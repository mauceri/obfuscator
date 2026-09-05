import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from rope_transform import sample_rope_rotation, sample_rope_scaling


def test_rope_rotation_is_orthogonal():
    d_head = 8
    r_hat = sample_rope_rotation(d_head, seed=0)
    assert r_hat.shape == (d_head, d_head)
    torch.testing.assert_close(r_hat @ r_hat.T, torch.eye(d_head), atol=1e-5, rtol=1e-5)


def test_rope_rotation_is_block_diagonal_2x2():
    d_head = 8
    r_hat = sample_rope_rotation(d_head, seed=1)
    for i in range(0, d_head, 2):
        for j in range(0, d_head, 2):
            if i != j:
                block = r_hat[i : i + 2, j : j + 2]
                assert torch.allclose(block, torch.zeros(2, 2), atol=1e-6)


def test_rope_scaling_is_diagonal_and_positive():
    d_head = 8
    h_hat = sample_rope_scaling(d_head, seed=2)
    assert h_hat.shape == (d_head, d_head)
    off_diag = h_hat - torch.diag(torch.diagonal(h_hat))
    assert torch.allclose(off_diag, torch.zeros(d_head, d_head))
    assert torch.all(torch.diagonal(h_hat) > 0)
