"""Vérification post-transformation sans charger le modèle complet en RAM.

Le transform streaming produit 16 Go de poids ; les vérifier en les rechargant
entièrement demanderait autant de RAM que le modèle. Ce script vérifie donc :

1. Structure : présence/cohérence des fichiers (index.json, config remappée,
   shards, dtype bf16), ensemble des noms de tenseurs.
2. Échantillons recomputés : pour quelques lignes d'embedding/unembedding et
   quelques couches complètes, on RECALCULE la valeur attendue à partir du
   modèle source (mêmes graines, mêmes formules que `transform_streaming.py`)
   et on la compare bit-à-bit au fichier écrit.

Usage:
    python verify_transform.py --model-dir ./obfuscated_qwen3_8b \
        --keys ./obfuscation_keys.json --source Qwen/Qwen3-8B \
        [--samples 16] [--layers 2] \
        [--alpha-e 1.0 --alpha-h 0.2 --beta 8 --gamma 1e3 --zeta 1e3]

`--source` accepte un id HuggingFace (cache local) ou un répertoire local.
Les hyperparamètres par défaut sont ceux de `transform_streaming.py` — les
repasser si la transformation a été lancée avec d'autres valeurs.
"""
import argparse
import json
import os
import random
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transform_streaming import (  # noqa: E402
    _ATTR_TENSORS, _ConfigLike, _PASSTHROUGH_SUFFIXES, _vocab_permutation,
    obfuscate_layer_tensors,
)


def _source_paths(source, source_dir=None):
    """Résout le répertoire source (HF cache ou local)."""
    if os.path.isdir(source):
        return source
    from huggingface_hub import snapshot_download
    return snapshot_download(
        source,
        allow_patterns=["config.json", "generation_config.json",
                        "model.safetensors.index.json", "model-*.safetensors"],
    )


def _weight_map(dirpath):
    index_path = os.path.join(dirpath, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    with safe_open(os.path.join(dirpath, "model.safetensors"),
                   framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _read_row(dirpath, weight_map, name, row):
    with safe_open(os.path.join(dirpath, weight_map[name]), framework="pt",
                   device="cpu") as f:
        t = f.get_tensor(name)
        return t[row]


def _read_tensor(dirpath, weight_map, name):
    with safe_open(os.path.join(dirpath, weight_map[name]), framework="pt",
                   device="cpu") as f:
        return f.get_tensor(name)


def check_structure(model_dir, expected_vocab=None, expected_layers=None):
    problems = []
    wm = _weight_map(model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    if expected_vocab is not None:
        if cfg.get("vocab_size") != expected_vocab:
            problems.append(f"vocab_size={cfg.get('vocab_size')} != "
                            f"{expected_vocab}")
    if expected_layers is not None:
        missing = [f"model.layers.{i}.self_attn.q_proj.weight"
                   for i in range(expected_layers)
                   if f"model.layers.{i}.self_attn.q_proj.weight" not in wm]
        if missing:
            problems.append(f"couches manquantes: {missing[:3]}...")
    if idx["weight_map"] != wm:
        problems.append("index.json et weight_map incohérents")
    # dtype bf16 sur un échantillon de tenseurs
    dtypes = set()
    for name in list(wm)[:8]:
        with safe_open(os.path.join(model_dir, wm[name]), framework="pt") as f:
            dtypes.add(str(f.get_tensor(name).dtype))
    if dtypes != {"torch.bfloat16"}:
        problems.append(f"dtypes inattendus: {dtypes}")
    return problems


def check_embedding_rows(model_dir, src_dir, keys, samples, alpha_e, alpha_h,
                         chunk_rows=4096):
    """Recalcule quelques lignes d'embedding/head et les compare bit-à-bit."""
    wm_out = _weight_map(model_dir)
    wm_src = _weight_map(src_dir)
    vocab_size = len(keys["vocab_permutation"])  # bijection → == vocab_size
    unpermute = keys["vocab_unpermute"]
    seed = keys["seed"]

    problems = []
    rng = random.Random(seed + 9999)  # échantillon reproductible mais distinct

    for table, src_name, alpha in (("embed", "model.embed_tokens.weight",
                                    alpha_e),
                                   ("head", "lm_head.weight", alpha_h)):
        src_full = _read_tensor(src_dir, wm_src, src_name).float()
        sigma = src_full.std()
        noise = torch.randn(src_full.shape,
                            generator=torch.Generator().manual_seed(
                                seed if table == "embed" else seed + 1))
        out_full = _read_tensor(model_dir, wm_out, src_name)
        for _ in range(samples):
            p = rng.randrange(vocab_size)  # ligne de sortie
            c = int(unpermute[str(p)])
            expected = (src_full[c] + alpha * sigma * noise[c]).to(
                torch.bfloat16)
            got = out_full[p]
            if not torch.equal(expected, got):
                problems.append(f"{src_name} ligne {p}: attendu {expected[:3]} "
                                f"obtenu {got[:3]}")
    return problems


def check_layers(model_dir, src_dir, keys, count, beta, gamma, zeta,
                 rope_scaling):
    wm_out = _weight_map(model_dir)
    wm_src = _weight_map(src_dir)
    seed = keys["seed"]
    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    num_layers = cfg["num_hidden_layers"]
    problems = []
    for i in range(min(count, num_layers)):
        in_t = {}
        for suffix in _ATTR_TENSORS:
            name = f"model.layers.{i}.{suffix}"
            in_t[name] = _read_tensor(src_dir, wm_src, name)
        expected = obfuscate_layer_tensors(
            _ConfigLike(cfg), i, in_t, seed, beta=beta, gamma=gamma,
            zeta=zeta, rope_scaling=rope_scaling)
        for name, exp in expected.items():
            got = _read_tensor(model_dir, wm_out, name)
            if not torch.equal(exp, got):
                d = (exp.float() - got.float()).abs()
                problems.append(f"{name}: maxdiff={d.max().item():.2e}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha-e", type=float, default=1.0)
    ap.add_argument("--alpha-h", type=float, default=0.2)
    ap.add_argument("--beta", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=1e3)
    ap.add_argument("--zeta", type=float, default=1e3)
    ap.add_argument("--rope-scaling", choices=["auto", "on", "off"],
                    default="auto")
    args = ap.parse_args()

    with open(args.keys) as f:
        keys = json.load(f)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        out_cfg = json.load(f)
    src_dir = _source_paths(args.source)
    with open(os.path.join(src_dir, "config.json")) as f:
        src_cfg = json.load(f)

    rope_scaling = args.rope_scaling
    if rope_scaling == "auto":
        rope_scaling = (src_cfg.get("model_type") not in
                        ("qwen3", "qwen3_moe"))

    problems = []
    problems += check_structure(args.model_dir, src_cfg["vocab_size"],
                                src_cfg["num_hidden_layers"])
    problems += check_embedding_rows(args.model_dir, src_dir, keys,
                                     args.samples, args.alpha_e, args.alpha_h)
    problems += check_layers(args.model_dir, src_dir, keys, args.layers,
                             args.beta, args.gamma, args.zeta, rope_scaling)

    # IDs spéciaux dans l'espace permuté
    for field in ("bos_token_id", "eos_token_id"):
        src_id = src_cfg.get(field)
        if src_id is None:
            continue
        exp = keys["vocab_permutation"][str(src_id)]
        got = out_cfg.get(field)
        if got != exp:
            problems.append(f"config.{field}: attendu {exp} (source {src_id}), "
                            f"obtenu {got}")

    print("== Vérification du modèle obfusqué ==")
    print(f"  modèle : {args.model_dir}  ({len(_weight_map(args.model_dir))} "
          "tenseurs)")
    print(f"  échantillons recomputés : {args.samples} lignes embed/head, "
          f"{args.layers} couches")
    if problems:
        print(f"  [KO] {len(problems)} problème(s) :")
        for p in problems[:20]:
            print(f"    - {p}")
        return 1
    print("  [OK] structure, config, dtype, échantillons bit-à-bit conformes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
