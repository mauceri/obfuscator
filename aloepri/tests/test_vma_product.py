"""Tests de la VMA PRODUIT (Table 9) — `vma_product.py`.

Verrouille le mécanisme de l'Appendice D dans sa forme complète : les
produits de poids annulent P̂/Q̂ (W̃_e·W̃_gateᵀ = Π·X·Ẑ_ffn), RowSort élimine
Ẑ_ffn, l'appariement des lignes triées récupère Π.

Mesures sur le toy chaîné (d=64, h=8) :
- α_e=0 : récupération 100 % (le mécanisme est exact) ;
- α_e=0.3 : ~62-69 % — le bruit d'embedding (seule défense corrélée) dégrade
  mais ne défend pas : la VMA produit est une vraie menace sur le schéma h>0.
"""
import torch

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ..chained_transform import obfuscate_chained
from ..vma_product import run_vma_product


def _tiny_qwen3(seed=7):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rope_theta=1e6,
        tie_word_embeddings=False, bos_token_id=62, eos_token_id=63,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).eval(), config


def _materials(alpha_e):
    toy, cfg = _tiny_qwen3()
    obf, keys = obfuscate_chained(toy, cfg, seed=3, alpha_e=alpha_e,
                                  alpha_h=0.0, h=8)
    obf_embed = obf.get_input_embeddings().weight.double()
    clear_embed = toy.get_input_embeddings().weight.double()
    obf_gates, clear_gates, wns = [], [], []
    for cl, ol in zip(toy.model.layers, obf.model.layers):
        obf_gates.append(ol.mlp.gate_proj.weight.double())
        clear_gates.append(cl.mlp.gate_proj.weight.double())
        wns.append(cl.post_attention_layernorm.weight.double())
    return (obf_embed, obf_gates, clear_embed, clear_gates, wns,
            keys["vocab_permutation"])


def test_vma_product_exact_no_noise():
    """α_e=0 : le produit + RowSort + appariement récupère Π à ~100 % —
    le mécanisme de la Table 9 fonctionne sur le schéma h>0."""
    mats = _materials(alpha_e=0.0)
    res = run_vma_product(*mats, subset_size=32, seed=1)
    assert res["taux_vote"] > 0.9, f"vote={res['taux_vote']:.1%}"
    assert all(r > 0.9 for r in res["taux_par_couche"])


def test_vma_product_noise_degrades_but_does_not_defend():
    """α_e=0.3 : la récupération reste très au-dessus du hasard (1/64) — le
    bruit d'embedding est la SEULE défense et elle ne suffit pas au toy."""
    mats = _materials(alpha_e=0.3)
    res = run_vma_product(*mats, subset_size=32, seed=1)
    assert res["taux_vote"] > 0.3, f"vote={res['taux_vote']:.1%}"


def test_vma_product_chance_baseline():
    """Sanity : un appariement aléatoire donnerait ~1/vocab (1.6 %) — les
    taux mesurés (62-100 %) sont bien un signal, pas du bruit."""
    assert 1 / 64 < 0.1
