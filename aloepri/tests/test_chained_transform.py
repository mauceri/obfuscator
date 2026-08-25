"""Tests du chaînage global P̂/Q̂ (h>0) — `chained_transform.py`.

Le round-trip est APPROXIMATIF par construction (correction κ par espérance,
§5.2.5) : on verrouille la structure — toute erreur dans P̂/Q̂, les folds
Wnorm ou κ fait tomber la corrélation de ~0.98 à ~0.1 (mesuré pendant le
développement). Le seuil corr > 0.95 distingue nettement les deux régimes.

Erreur κ ~ CV(‖xP̂‖/‖x‖) ≈ 1/√(h/2) : ~26 % à h=8 (toy), ~9 % à h=128
(réglage du papier). Le toy est donc un test PLUS sévère que le 8B.
"""

import copy
import math

import numpy as np
import pytest
import torch

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ..chained_transform import estimate_kappa, obfuscate_chained
from ..key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen


def _tiny_qwen3(seed=0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rope_theta=1e6,
        tie_word_embeddings=False, bos_token_id=62, eos_token_id=63,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).eval(), config


def _round_trip(model, config, obf, perm, ids):
    """Logits obfusqués (permutés + dépermutés) vs logits clairs."""
    permuted = torch.tensor([[perm[int(t)] for t in ids[0]]])
    with torch.no_grad():
        baseline = model(ids).logits.double()
        obs = obf(permuted).logits.double()
    cols = torch.tensor([perm[t] for t in range(config.vocab_size)])
    obs_p = obs[..., cols]
    err = ((obs_p - baseline).abs().max().item()
           / baseline.abs().max().item())
    corr = torch.stack([obs_p.flatten(), baseline.flatten()]
                       ).corrcoef()[0, 1].item()
    return err, corr


def test_kappa_orthogonal_h0():
    """h=0, λ=0 : P̂ orthogonal (B=U) → ‖xP̂‖=‖x‖ → κ = √(d/d) = 1 exact."""
    rng = np.random.default_rng(0)
    base = init_key_matrix(16, 0, 0.0, rng)
    p_hat = key_mat_gen(base)
    kappa = estimate_kappa(p_hat, n=4096)
    assert abs(kappa - 1.0) < 1e-3


def test_kappa_matches_monte_carlo_definition():
    """κ = E[‖xP̂‖/‖x‖]·√(d/(d+2h)) — la fonction calcule bien la formule."""
    d, h = 64, 8
    rng = np.random.default_rng(3)
    base = init_key_matrix(d, h, 0.3, rng)
    p_hat = key_mat_gen(base)
    kappa = estimate_kappa(p_hat, n=8192, seed=1)
    P = torch.tensor(p_hat, dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8192, d, generator=g, dtype=torch.float64)
    expected = ((x @ P).norm(dim=1) / x.norm(dim=1)).mean()
    expected = float(expected * math.sqrt(d / P.shape[1]))
    assert abs(kappa - expected) < 1e-9


def test_key_matrices_exact_inverses_h():
    """P̂ (d, d+2h) · Q̂ (d+2h, d) = I_d — invariant du chaînage."""
    for (seed, d, h) in [(0, 64, 8), (1, 64, 32)]:
        rng = np.random.default_rng(seed)
        base = init_key_matrix(d, h, 0.3, rng)
        p_hat = key_mat_gen(base)
        q_hat = inv_key_mat_gen(base)
        assert p_hat.shape == (d, d + 2 * h)
        assert q_hat.shape == (d + 2 * h, d)
        np.testing.assert_allclose(p_hat @ q_hat, np.eye(d), atol=1e-8)


def test_chained_roundtrip_toy():
    """Round-trip complet (embed→head, 2 couches) : corr > 0.95 à h=8.

    Toute erreur de structure (P̂/Q̂, folds Wnorm, κ, permutation embed/head)
    fait tomber la corrélation à ~0.1 (observé en développement)."""
    torch.manual_seed(7)
    model, config = _tiny_qwen3(7)
    ids = torch.tensor([[1, 7, 13, 42, 5, 60]])
    obf, keys = obfuscate_chained(
        model, config, seed=3, alpha_e=0.0, alpha_h=0.0, h=8)
    assert obf.config.hidden_size == config.hidden_size + 16
    err, corr = _round_trip(model, config, obf, keys["vocab_permutation"], ids)
    assert corr > 0.95, f"round-trip h>0 cassé : corr={corr:.4f} err={err:.4f}"
    assert err < 0.6


def test_chained_roundtrip_with_noise():
    """Avec le bruit α_e=0.3/α_h=0.2 (config production), la corrélation
    reste élevée — le bruit s'ajoute à l'approximation κ, ne la domine pas."""
    torch.manual_seed(7)
    model, config = _tiny_qwen3(7)
    ids = torch.tensor([[1, 7, 13, 42, 5, 60]])
    obf, keys = obfuscate_chained(
        model, config, seed=3, alpha_e=0.3, alpha_h=0.2, h=8)
    err, corr = _round_trip(model, config, obf, keys["vocab_permutation"], ids)
    assert corr > 0.9, f"round-trip bruité cassé : corr={corr:.4f}"


def test_chained_embedding_defends_vma_direct():
    """La VMA directe (embedding vs table claire) devient IMPOSSIBLE :
    dimensions d+2h ≠ d. C'est la défense structurelle que h=0 n'avait pas."""
    torch.manual_seed(7)
    model, config = _tiny_qwen3(7)
    obf, _ = obfuscate_chained(model, config, seed=3, h=8)
    obf_shape = obf.get_input_embeddings().weight.shape
    clear_shape = model.get_input_embeddings().weight.shape
    assert obf_shape == (config.vocab_size, config.hidden_size + 16)
    assert obf_shape[1] != clear_shape[1]
    # et le produit W̃_e·W̃_q annule bien P̂ : les produits restent comparables
    # (vue Table 9) — vérifié via l'invariant P̂Q̂=I en test_key_matrices.


def test_chained_keys_match_poc_permutation():
    """Même seed → même permutation de vocabulaire que le POC h=0 : les clés
    locales existantes restent valides pour le modèle chaîné."""
    torch.manual_seed(7)
    model, config = _tiny_qwen3(7)
    _, keys = obfuscate_chained(model, config, seed=0, h=8)
    # même tirage que embedding_obfuscation (random.Random(0).shuffle)
    import random as _random
    rng = _random.Random(0)
    expected = list(range(config.vocab_size))
    rng.shuffle(expected)
    assert keys["vocab_permutation"] == dict(
        zip(range(config.vocab_size), expected))
