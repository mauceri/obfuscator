"""Tests des attaques ISA par gradient (`isa_attack.py`) sur modèle Qwen3
miniature.

Ce qui est robuste à cette échelle (et verrouillé ici) :
1. Canal HIDDEN : l'inversion par descente de gradient récupère à ~100 % les
   IDs réellement envoyés au modèle — sur la baseline (IDs clairs) comme sur
   le modèle obfusqué (IDs PERMUTÉS : c'est l'entrée que le modèle a vue).
   Sur l'obfusqué, les IDs récupérés ne sont donc PAS le texte clair : sans
   la clé de permutation (côté client), l'attaquant ne lit rien.
2. Canal ATTN : la machinerie tourne et la loss décroît (sur modèle jouet le
   canal est sous-déterminé — l'évaluation quantitative se fait en grandeur
   nature, cf. `aloepri_modal/app.py::isa_attack`).
"""
import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from isa_attack import run_channel_attack
from model_transform import obfuscate_model_in_place

transformers = pytest.importorskip("transformers")


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


def _secret_ids(seed=123, length=8):
    torch.manual_seed(seed)
    return torch.randint(5, 60, (length,)).tolist()


def test_hidden_channel_recovers_baseline_input_ids():
    """Canal hidden, baseline : l'attaque récupère les IDs clairs envoyés."""
    model, _ = _tiny_qwen3()
    secret = _secret_ids()
    pred, rate, _ = run_channel_attack(model, secret, "hidden", 1,
                                       steps=600, seed=0, device="cpu")
    assert rate >= 0.9, f"récupération baseline insuffisante: {rate:.1%}"
    assert pred.tolist() == secret


def test_hidden_channel_on_obfuscated_recovers_permuted_ids_not_text():
    """Canal hidden, modèle obfusqué : l'attaque récupère les IDs PERMUTÉS
    (l'entrée réelle du modèle), pas le texte clair — la permutation est le
    secret que l'attaquant serveur ne possède pas."""
    model, config = _tiny_qwen3()
    keys = obfuscate_model_in_place(model, config, seed=7, alpha_e=0.5,
                                    alpha_h=0.1, beta=1)
    secret = _secret_ids()
    permuted = [keys.vocab_permutation[i] for i in secret]
    assert permuted != secret, "la permutation doit déplacer au moins un id"

    pred, rate, _ = run_channel_attack(model, permuted, "hidden", 1,
                                       steps=600, seed=0, device="cpu")
    assert rate >= 0.9, f"récupération des ids permutés insuffisante: {rate:.1%}"
    assert pred.tolist() != secret, (
        "les ids récupérés ne doivent PAS être le texte clair (pas de clé)")


def test_attn_channel_runs_and_loss_decreases():
    """Canal attn : la machinerie tourne et la loss décroît. (Sur modèle
    jouet le canal est sous-déterminé — pas d'assertion de taux ici.)"""
    model, _ = _tiny_qwen3()
    secret = _secret_ids()
    pred, rate, losses = run_channel_attack(model, secret, "attn", 0,
                                            steps=300, seed=0, device="cpu")
    assert len(pred) == len(secret)
    assert losses[-1] < losses[0], "la loss d'inversion doit décroître"
    assert losses[-1] < 0.5, f"loss relative finale trop haute: {losses[-1]:.4f}"
