"""Tests d'intégration : l'obfuscation d'attention face à la VRAIE RoPE de
HuggingFace/Qwen2, puis le pipeline complet (`model_transform`) sur un modèle
Qwen2 miniature.

Les tests de la Task 7 (`test_attention_obfuscation.py`) vérifient le
round-trip *sans* RoPE et *sans* biais : ils ne peuvent donc pas attraper
(a) la convention RoPE (le papier apparie (2i, 2i+1), HF apparie (i, i+d/2)),
ni (b) les biais q/k/v de Qwen2, qui doivent subir la même transformation que
les poids. Ce fichier ferme les deux trous.
"""
import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from attention_obfuscation import obfuscate_attention_layer

transformers = pytest.importorskip("transformers")
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from model_transform import obfuscate_model_in_place


# --------------------------------------------------------------------------
# 1. Attention + RoPE réelle (rotate_half)
# --------------------------------------------------------------------------

def _rope(d_head, seq_len, theta=1e6):
    """cos/sin produits par l'implémentation HF elle-même, pas par une
    ré-écriture locale : c'est la convention de HF qu'on veut tester."""
    config = Qwen2Config(hidden_size=d_head, num_attention_heads=1, rope_theta=theta)
    rotary = Qwen2RotaryEmbedding(config)
    dummy = torch.zeros(1, seq_len, d_head)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    return rotary(dummy, position_ids)


def _attention_with_rope(x, w_q, w_k, w_v, w_o, b_q, b_k, b_v,
                         num_heads, num_kv_heads, d_head):
    """GQA + RoPE HF, écrite à la main pour rester lisible et indépendante des
    optimisations internes de `Qwen2Attention`."""
    seq_len = x.shape[0]
    group_size = num_heads // num_kv_heads
    cos, sin = _rope(d_head, seq_len)

    q = (x @ w_q.T + b_q).view(seq_len, num_heads, d_head).transpose(0, 1)
    k = (x @ w_k.T + b_k).view(seq_len, num_kv_heads, d_head).transpose(0, 1)
    v = (x @ w_v.T + b_v).view(seq_len, num_kv_heads, d_head).transpose(0, 1)
    q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q, k = q[0], k[0]

    outputs = []
    for h in range(num_heads):
        kv = h // group_size
        weights = torch.softmax(q[h] @ k[kv].T / d_head**0.5, dim=-1)
        outputs.append(weights @ v[kv])
    return torch.cat(outputs, dim=-1) @ w_o.T


def _layer(hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed):
    torch.manual_seed(seed)
    return dict(
        w_q=torch.randn(num_heads * d_head, hidden_size) * 0.1,
        w_k=torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        w_v=torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        w_o=torch.randn(hidden_size, num_heads * d_head) * 0.1,
        b_q=torch.randn(num_heads * d_head) * 0.1,
        b_k=torch.randn(num_kv_heads * d_head) * 0.1,
        b_v=torch.randn(num_kv_heads * d_head) * 0.1,
    )


GEOM = dict(hidden_size=64, num_heads=8, num_kv_heads=2, d_head=32)


def _obfuscate(w, rope_layout, beta=1, with_bias=True, seed=0):
    return obfuscate_attention_layer(
        w["w_q"], w["w_k"], w["w_v"], w["w_o"],
        num_heads=GEOM["num_heads"], num_kv_heads=GEOM["num_kv_heads"],
        d_head=GEOM["d_head"], beta=beta, gamma=1e3, zeta=1e3, seed=seed,
        b_q=w["b_q"] if with_bias else None,
        b_k=w["b_k"] if with_bias else None,
        b_v=w["b_v"] if with_bias else None,
        rope_layout=rope_layout,
    )


def _error(w, obf, x):
    """Erreur relative de la sortie d'attention obfusquée vs claire."""
    baseline = _attention_with_rope(
        x, w["w_q"], w["w_k"], w["w_v"], w["w_o"], w["b_q"], w["b_k"], w["b_v"],
        GEOM["num_heads"], GEOM["num_kv_heads"], GEOM["d_head"],
    )
    got = _attention_with_rope(
        x, obf.w_q_obf, obf.w_k_obf, obf.w_v_obf, obf.w_o_obf,
        obf.b_q_obf if obf.b_q_obf is not None else w["b_q"],
        obf.b_k_obf if obf.b_k_obf is not None else w["b_k"],
        obf.b_v_obf if obf.b_v_obf is not None else w["b_v"],
        GEOM["num_heads"], GEOM["num_kv_heads"], GEOM["d_head"],
    )
    return ((got - baseline).abs().max() / baseline.abs().max()).item()


def test_attention_round_trip_survives_hf_rope():
    """Avec `rope_layout="half"` (conjugaison par π) et β=1 — c.-à-d. Ẑ_block =
    identité, donc sans le mélange de fréquences que le papier assume comme
    approximation — la reparamétrisation reste EXACTE à travers la rotation
    RoPE de HuggingFace."""
    x = torch.randn(6, GEOM["hidden_size"], generator=torch.Generator().manual_seed(99))
    for seed in (0, 1, 2):
        w = _layer(**GEOM, seq_len=6, seed=seed)
        err = _error(w, _obfuscate(w, "half", seed=seed), x)
        assert err < 1e-4, f"round-trip cassé sous RoPE HF (seed={seed}) : {err}"


def test_paper_interleaved_layout_is_destroyed_by_hf_rope():
    """Contrôle négatif n°1 — sans la conjugaison par π, le test précédent
    devient faux : les facteurs R̂/Ĥ/Ẑ construits sur les paires (2i, 2i+1) du
    papier ne commutent pas avec `rotate_half`, qui apparie (i, i+d/2).
    Sans ce test, une conjugaison omise ou inversée passerait inaperçue."""
    x = torch.randn(6, GEOM["hidden_size"], generator=torch.Generator().manual_seed(99))
    w = _layer(**GEOM, seq_len=6, seed=0)
    err = _error(w, _obfuscate(w, "interleaved"), x)
    assert err > 0.1, f"la convention du papier devrait casser sous RoPE HF, erreur={err}"


def test_untransformed_biases_break_the_round_trip():
    """Contrôle négatif n°2 — les biais q/k/v de Qwen2 (bias=True, contrairement
    à Llama) doivent subir le même facteur que les poids. Laissés tels quels,
    ils cassent l'invariance."""
    x = torch.randn(6, GEOM["hidden_size"], generator=torch.Generator().manual_seed(99))
    w = _layer(**GEOM, seq_len=6, seed=0)
    err = _error(w, _obfuscate(w, "half", with_bias=False), x)
    assert err > 0.1, f"des biais non transformés devraient casser le round-trip, erreur={err}"


# --------------------------------------------------------------------------
# 2. Pipeline complet sur un Qwen2 miniature
# --------------------------------------------------------------------------

def _tiny_model(seed=0, tie_word_embeddings=False):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rope_theta=1e6,
        tie_word_embeddings=tie_word_embeddings,
        # dans le vocabulaire du modèle jouet (les défauts de Qwen2Config sont
        # ceux du vrai modèle, ~151k, hors des 64 lignes d'ici)
        bos_token_id=62, eos_token_id=63, pad_token_id=None,
    )
    model = Qwen2ForCausalLM(config).eval()
    # `_init_weights` de HF met les biais à zéro : des biais nuls rendraient
    # leur transformation invisible. On les remplit donc explicitement.
    for layer in model.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.bias.data.normal_(0.0, 0.1)
    return model, config


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_tiny_qwen2_logits_are_preserved_up_to_the_vocabulary_permutation(
    tie_word_embeddings,
):
    """Le POC complet monté bout en bout : embedding + attention + FFN
    obfusqués sur un vrai `Qwen2ForCausalLM` (miniature). Le serveur reçoit des
    IDs permutés et doit renvoyer les mêmes logits, eux aussi permutés.

    β=1 (Ẑ_block = identité) et α=0 : on isole ici la CORRECTION de la
    reparamétrisation, pas la dégradation volontaire (bruit d'embedding,
    mélange de fréquences RoPE) que la Task 9 mesurera sur le vrai modèle.

    Les deux régimes de weight tying sont couverts : Qwen2.5-7B-Instruct n'est
    pas lié, mais les petits Qwen2.5 le sont, et la branche `tied` de
    `obfuscate_model_in_place` (ne pas écrire deux fois dans le même tenseur)
    ne serait sinon jamais exécutée."""
    model, config = _tiny_model(tie_word_embeddings=tie_word_embeddings)
    clear_ids = torch.tensor([[1, 7, 13, 42, 5, 5, 60]])
    with torch.no_grad():
        baseline = model(clear_ids).logits

    keys = obfuscate_model_in_place(
        model, config, seed=3, alpha_e=0.0, alpha_h=0.0, beta=1,
    )

    permuted_ids = torch.tensor([[keys.vocab_permutation[int(t)] for t in clear_ids[0]]])
    with torch.no_grad():
        obfuscated = model(permuted_ids).logits

    # colonne `permutation[t]` des logits obfusqués == colonne `t` des logits clairs
    columns = torch.tensor([keys.vocab_permutation[t] for t in range(config.vocab_size)])
    torch.testing.assert_close(obfuscated[..., columns], baseline, atol=1e-4, rtol=1e-3)


def test_special_token_ids_are_remapped_into_the_permuted_space(tmp_path):
    """Le modèle obfusqué ÉMET des IDs permutés. Un `eos_token_id` laissé en
    clair ne serait donc jamais produit — `generate()` ne s'arrêterait jamais —
    et l'ID clair de l'EOS serait émis à la place d'un token banal, provoquant
    un arrêt prématuré. Le test passe par un vrai `save_pretrained` /
    `from_pretrained` : ce sont les fichiers livrés au serveur qui comptent, pas
    l'objet en mémoire."""
    model, config = _tiny_model()
    bos_clair, eos_clair = config.bos_token_id, config.eos_token_id

    keys = obfuscate_model_in_place(model, config, seed=3, alpha_e=0.0, alpha_h=0.0, beta=1)
    model.save_pretrained(tmp_path)
    relu = Qwen2ForCausalLM.from_pretrained(tmp_path)

    assert relu.config.eos_token_id == keys.vocab_permutation[eos_clair]
    assert relu.config.bos_token_id == keys.vocab_permutation[bos_clair]
    assert relu.config.eos_token_id != eos_clair  # la permutation bouge cet ID
    assert relu.generation_config.eos_token_id == keys.vocab_permutation[eos_clair]
    assert relu.config.pad_token_id is None  # un champ absent le reste


def test_tiny_qwen2_weights_actually_changed():
    """Garde-fou : le test précédent passerait aussi si `obfuscate_model_in_place`
    ne faisait rien du tout (la permutation du vocabulaire étant alors la seule
    différence). On vérifie donc que les poids internes ont bougé."""
    model, config = _tiny_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    obfuscate_model_in_place(model, config, seed=3, alpha_e=0.0, alpha_h=0.0, beta=1)

    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ):
        after = dict(model.named_parameters())[name]
        assert not torch.allclose(before[name], after), f"{name} n'a pas été obfusqué"
