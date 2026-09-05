"""Rotation RoPE 2D par paire + scaling (Algorithme 2, lignes 1-2)."""
import torch


def sample_rope_rotation(d_head, seed):
    assert d_head % 2 == 0
    gen = torch.Generator().manual_seed(seed)
    n_pairs = d_head // 2
    rho = torch.rand(n_pairs, generator=gen) * 2 * torch.pi

    r_hat = torch.zeros(d_head, d_head)
    for i, angle in enumerate(rho):
        c, s = torch.cos(angle), torch.sin(angle)
        r_hat[2 * i, 2 * i] = c
        r_hat[2 * i, 2 * i + 1] = -s
        r_hat[2 * i + 1, 2 * i] = s
        r_hat[2 * i + 1, 2 * i + 1] = c
    return r_hat


def sample_rope_scaling(d_head, seed):
    assert d_head % 2 == 0
    gen = torch.Generator().manual_seed(seed)
    n_pairs = d_head // 2
    # strictement positif pour rester inversible
    s = torch.exp(torch.randn(n_pairs, generator=gen) * 0.1)
    diag = s.repeat_interleave(2)
    return torch.diag(diag)
