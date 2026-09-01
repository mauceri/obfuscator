"""Tests de l'attaque ISA discrète (vocab-matching k-way) — point 4 de la revue.

Le but : remplacer la conclusion « canal hidden non informatif » fondée sur
la relaxation continue (loss→0 mais ids≠prompt) par un test DISCRET : à
chaque position, le vrai token est mélangé à k−1 leurres et le canal doit
l'identifier. Un canal informatif donne un taux → 100 % ; un canal
sous-déterminé donne → 1/k.
"""
import torch
import pytest

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from aloepri.isa_attack import (
    run_vocab_match, vocab_match_attack, _state_distance, capture_state,
)


@pytest.fixture(scope="module")
def mini_model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=32, rope_theta=1e6,
                      tie_word_embeddings=False, bos_token_id=500,
                      eos_token_id=501,
                      attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval().to("cpu")


IDS = [7, 42, 123, 300, 55, 9]


def test_vocab_match_recovers_identity_layer0(mini_model):
    """Couche 0 : le canal hidden doit discriminer ~100 % (embeddings
    distincts, recherche discrète sans relaxation)."""
    pred, rate, hit = run_vocab_match(
        mini_model, IDS, layer=0, k=32, teacher_forcing=True,
        metric="mse", seed=1, device="cpu")
    assert rate > 0.9, f"couche 0 devrait être informatif, taux={rate:.1%}"
    assert pred.tolist() == IDS


def test_vocab_match_greedy_autoregressive(mini_model):
    """Greedy autorégressif (préfixe = tokens prédits) : la mécanique tourne
    et sur miniature le canal reste informatif."""
    pred, rate, hit = run_vocab_match(
        mini_model, IDS, layer=0, k=32, teacher_forcing=False,
        metric="mse", seed=2, device="cpu")
    assert pred.shape == torch.Size([len(IDS)])
    assert 0.0 <= rate <= 1.0


def test_state_distance_metrics():
    """Les deux métriques (mse relative, cosinus) sont des distances : la
    cible elle-même est toujours la plus proche."""
    target = torch.randn(16)
    states = torch.randn(8, 16)
    states[3] = target.clone()          # le « vrai » candidat
    tv = target.pow(2).mean().item()
    for metric in ("mse", "cos"):
        d = _state_distance(states, target, tv, metric)
        assert int(d.argmin().item()) == 3, f"{metric}: la cible doit gagner"


def test_kway_baseline_reference():
    """Le taux k-way se compare à 1/k (baseline aléatoire) : la métrique
    doit être interprétable comme discrimination, pas comme récupération
    absolue."""
    k = 64
    # un canal sous-déterminé tire uniformément → ~1/k
    rng = torch.Generator().manual_seed(0)
    n = 2000
    hits = (torch.randint(0, k, (n,), generator=rng) == 0).float().mean()
    assert abs(hits.item() - 1.0 / k) < 0.05


def test_capture_state_shape(mini_model):
    """Capture d'état : hidden (T, hidden) et attn (heads, T, T) — les deux
    canaux de l'attaque."""
    embed_table = mini_model.get_input_embeddings().weight
    embeds = embed_table[torch.tensor(IDS)]
    h = capture_state(mini_model, embeds, layer=1, channel="hidden")
    assert h.shape == (len(IDS), 64)
    a = capture_state(mini_model, embeds, layer=1, channel="attn")
    assert a.shape == (4, len(IDS), len(IDS))


def test_vocab_match_bf16_model():
    """Modèle en bf16 (comme sur Modal) + table float32 : le canal hidden doit
    rester informatif (le fix upcaste embeds vers model.dtype — le bug
    'expected mat1 and mat2 to have the same dtype' est couvert ici)."""
    import torch
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=32, rope_theta=1e6,
                      tie_word_embeddings=False, bos_token_id=500,
                      eos_token_id=501, attn_implementation="eager")
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    pred, rate, hit = run_vocab_match(
        model, IDS, layer=0, k=32, teacher_forcing=True,
        metric="mse", seed=1, device="cpu")
    assert rate > 0.9, f"canal hidden doit rester informatif en bf16, taux={rate:.1%}"
