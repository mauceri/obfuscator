"""Tests d'intégration Qwen3 : la reparamétrisation doit survivre aux normes
de tête q_norm/k_norm (→ rope_scaling=off automatique), et le transform en
streaming mémoire-léger doit produire des poids bit-à-bit identiques au
transform en place testé (`model_transform.obfuscate_model_in_place`).

Qwen3 diffère de Qwen2.5 sur un point qui touche le cœur de l'obfuscation
d'attention : `q_norm`/`k_norm`, une RMSNorm PAR TÊTE appliquée à q/k juste
après la projection, avant RoPE (`query_states = self.q_norm(self.q_proj(...))`
dans `modeling_qwen3.py`). Une RMSNorm de tête ne commute qu'avec les facteurs
orthogonaux (R̂, Ẑ) ; le scaling diagonal Ĥ du papier casse le round-trip.
Ces tests verrouillent la détection automatique et l'équivalence des deux
pipelines de transformation.
"""
import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from model_transform import obfuscate_model_in_place
from transform_streaming import obfuscate_embedding_chunked, transform_streaming

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM


def _tiny_qwen3(seed=0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, rope_theta=1e6,
        tie_word_embeddings=False,
        bos_token_id=62, eos_token_id=63,
    )
    return Qwen3ForCausalLM(config).eval(), config


def _roundtrip_error(rope_scaling, seed=3):
    """Erreur relative max entre les logits baseline et les logits obfusqués
    (réordonnés par la permutation de vocabulaire)."""
    model, config = _tiny_qwen3(seed=7)
    clear_ids = torch.tensor([[1, 7, 13, 42, 5, 60]])
    with torch.no_grad():
        baseline = model(clear_ids).logits

    keys = obfuscate_model_in_place(
        model, config, seed=seed, alpha_e=0.0, alpha_h=0.0, beta=1,
        rope_scaling=rope_scaling,
    )
    permuted = torch.tensor(
        [[keys.vocab_permutation[int(t)] for t in clear_ids[0]]])
    with torch.no_grad():
        obfuscated = model(permuted).logits

    columns = torch.tensor(
        [keys.vocab_permutation[t] for t in range(config.vocab_size)])
    diff = (obfuscated[..., columns] - baseline).abs()
    return (diff.max() / baseline.abs().max()).item()


def test_tiny_qwen3_logits_preserved_up_to_vocab_permutation():
    """Le pipeline complet sur un vrai `Qwen3ForCausalLM` (miniature) : avec
    la détection automatique (q_norm présent → rope_scaling=off), les logits
    sont préservés à travers q_norm + RoPE HF, comme sur Qwen2.5."""
    err = _roundtrip_error(rope_scaling=None)
    assert err < 1e-3, f"round-trip Qwen3 cassé en auto : {err}"


def test_tiny_qwen3_rope_scaling_on_breaks_the_round_trip():
    """Contrôle négatif : si on force Ĥ (rope_scaling=on) sur Qwen3, la
    RMSNorm de tête casse la reparamétrisation — l'erreur saute de plusieurs
    ordres de grandeur. Ce test verrouille POURQUOI l'auto-détection existe."""
    err_auto = _roundtrip_error(rope_scaling=None)
    err_on = _roundtrip_error(rope_scaling=True)
    assert err_on > 1e-2, f"Ĥ devrait casser le round-trip Qwen3 : {err_on}"
    assert err_on > 100 * err_auto, (
        f"forcer Ĥ ne devrait pas être anodin (auto={err_auto}, on={err_on})")


def test_chunked_embedding_noise_matches_full_tensor_draw():
    """Le tirage du bruit par blocs de lignes (transform_streaming) consomme
    le même flux de `randn` que le tirage complet (transform en place)."""
    vocab, d = 1000, 64
    gen = torch.Generator().manual_seed(123)
    full = torch.randn(vocab, d, generator=gen)

    gen2 = torch.Generator().manual_seed(123)
    chunks = torch.cat([
        torch.randn(256, d, generator=gen2),
        torch.randn(256, d, generator=gen2),
        torch.randn(256, d, generator=gen2),
        torch.randn(232, d, generator=gen2),
    ])
    assert torch.equal(full, chunks)


def test_tiny_qwen3_streaming_equals_inplace(tmp_path):
    """Les deux pipelines de transformation (en place, chargé entièrement ;
    streaming, shard par shard) produisent des poids, des clés et des
    configs IDENTIQUES sur le même modèle miniature — le streaming ne doit
    être qu'une optimisation mémoire, pas une autre transformation."""
    src, ref, out = tmp_path / "src", tmp_path / "ref", tmp_path / "out"

    # source (poids d'origine, en bf16 comme le vrai checkpoint Qwen3)
    model, config = _tiny_qwen3(seed=7)
    model.to(torch.bfloat16)
    model.save_pretrained(src)

    # référence : transform en place (chemin testé par test_model_transform)
    model2, config2 = _tiny_qwen3(seed=7)
    model2.to(torch.bfloat16)
    keys_ref = obfuscate_model_in_place(
        model2, config2, seed=3, alpha_e=0.5, alpha_h=0.1, beta=8,
        gamma=1e3, zeta=1e3,
    )
    model2.save_pretrained(ref)

    # streaming
    keys_stream = transform_streaming(
        str(src), str(out), seed=3, alpha_e=0.5, alpha_h=0.1, beta=8,
        gamma=1e3, zeta=1e3, keys_path=str(tmp_path / "keys.json"),
    )

    # poids bit-à-bit identiques
    a = AutoModelForCausalLM.from_pretrained(ref, dtype=torch.bfloat16)
    b = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    differing = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not differing, f"tenseurs différents: {differing}"

    # clés identiques
    assert keys_ref.vocab_permutation == keys_stream.vocab_permutation
    assert keys_ref.vocab_unpermute == keys_stream.vocab_unpermute

    # IDs spéciaux remappés dans l'espace permuté (même valeur dans les deux)
    assert b.config.eos_token_id == a.config.eos_token_id
    assert b.config.eos_token_id == keys_ref.vocab_permutation[63]
    assert b.config.bos_token_id == keys_ref.vocab_permutation[62]

    # le streaming ne laisse aucun tenseur de couche orphelin (l'index couvre
    # tout, et rien d'extra n'est écrit)
    import json
    with open(out / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    assert set(wm) == set(sa)


def test_multi_shard_output_equals_inplace(tmp_path):
    """Le writer multi-shards (petit `shard_target_bytes` force plusieurs
    fichiers) produit des shards correctement nommés `model-XXXXX-of-NNNNN`,
    un index qui se charge, et des poids identiques au transform en place —
    le découpage en shards ne doit rien changer aux valeurs."""
    import json

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=384,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rope_theta=1e6,
        tie_word_embeddings=False, bos_token_id=500, eos_token_id=511,
    )
    src, out = tmp_path / "src", tmp_path / "out"
    model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16)
    model.save_pretrained(src)

    transform_streaming(str(src), str(out), seed=5, alpha_e=0.7, alpha_h=0.2,
                        beta=8, gamma=1e3, zeta=1e3,
                        keys_path=str(tmp_path / "k.json"),
                        shard_target_bytes=300_000)

    shards = sorted(f.name for f in out.iterdir()
                    if f.name.startswith("model-")
                    and f.name.endswith(".safetensors"))
    assert len(shards) > 1, "le test doit forcer plusieurs shards"
    totals = {int(s.rsplit("-of-", 1)[1].split(".")[0]) for s in shards}
    assert totals == {len(shards)}, f"numérotation incohérente: {shards}"
    with open(out / "model.safetensors.index.json") as f:
        assert set(json.load(f)["weight_map"]) == set(model.state_dict())

    # référence in-place (même graine d'init → mêmes poids source)
    torch.manual_seed(11)
    model2 = Qwen3ForCausalLM(config).eval().to(torch.bfloat16)
    obfuscate_model_in_place(model2, config, seed=5, alpha_e=0.7, alpha_h=0.2,
                             beta=8, gamma=1e3, zeta=1e3)
    loaded = AutoModelForCausalLM.from_pretrained(str(out),
                                                  dtype=torch.bfloat16)
    sa, sb = model2.state_dict(), loaded.state_dict()
    differing = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not differing, f"tenseurs différents (multi-shard): {differing}"


def test_embedding_chunked_matches_reference_math():
    """`obfuscate_embedding_chunked` reproduit la formule du POC
    (W* = W + α·σ(W)·bruit, lignes permutées par Π) — vérifié sur des valeurs
    arbitraires contre une implémentation de référence en une passe."""
    vocab, d = 128, 32
    torch.manual_seed(5)
    w_embed = torch.randn(vocab, d).to(torch.bfloat16)
    w_head = torch.randn(vocab, d).to(torch.bfloat16)

    # référence en une passe (même formule qu'embedding_obfuscation)
    import random
    rng = random.Random(11)
    permuted = list(range(vocab))
    rng.shuffle(permuted)
    unpermute = {v: k for k, v in enumerate(permuted)}
    inv = torch.tensor([unpermute[i] for i in range(vocab)])
    e32, h32 = w_embed.float(), w_head.float()
    noise_e = torch.randn(vocab, d, generator=torch.Generator().manual_seed(11))
    noise_h = torch.randn(vocab, d, generator=torch.Generator().manual_seed(12))
    ref_e = (e32 + 0.7 * e32.std() * noise_e)[inv].to(torch.bfloat16)
    ref_h = (h32 + 0.3 * h32.std() * noise_h)[inv].to(torch.bfloat16)

    got_e, got_h, perm, unperm, _, _ = obfuscate_embedding_chunked(
        w_embed, w_head, 0.7, 0.3, 11, chunk_rows=37)
    assert torch.equal(got_e, ref_e)
    assert torch.equal(got_h, ref_h)
    assert perm == {i: permuted[i] for i in range(vocab)}
    assert unperm == {v: k for k, v in enumerate(permuted)}
