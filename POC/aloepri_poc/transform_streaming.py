"""Transformation AloePri en streaming mémoire-léger (variante de
`model_transform.py`).

Problème résolu
---------------
`model_transform.py` charge le modèle complet en RAM (Qwen3-8B : ~16 Go en
bfloat16) puis applique l'obfuscation ; son pic mémoire est d'environ 30 Go
(modèle + copies float32 de l'embedding). Ce script produit STRICTEMENT les
mêmes poids obfusqués (mêmes graines, mêmes fonctions de couche) en ne
chargeant jamais plus d'un shard de poids à la fois : pic ≈ 4-6 Go, quel que
soit le modèle. Il tourne donc sur une machine de bureau (ou un petit worker
Modal) là où la version chargée entièrement exigerait une instance à 32 Go+.

Garanties d'équivalence
-----------------------
- Embedding/unembedding : le bruit et la permutation sont des opérations par
  ligne. Le bruit est tiré par blocs de lignes consécutives avec le MÊME
  générateur torch (`manual_seed(seed)` / `seed+1`) que
  `embedding_obfuscation.obfuscate_embedding` : le flux de `randn` est
  identique, donc les valeurs sont bit-à-bit identiques (vérifié dans
  `tests/test_qwen3_arch.py::test_tiny_qwen3_streaming_equals_inplace`).
- Attention/FFN : appels EXACTS des fonctions testées
  (`obfuscate_attention_layer`, `obfuscate_ffn_layer`) avec les mêmes
  graines, sur les mêmes tenseurs bf16 → mêmes résultats.
- `rope_scaling` : même logique 'auto' que `model_transform.py` (off pour les
  architectures à q_norm/k_norm type Qwen3, on sinon), exposée en CLI.

Usage:
    python transform_streaming.py --model Qwen/Qwen3-8B \\
        --output ./obfuscated_qwen3_8b --keys ./obfuscation_keys.json --seed 0

`--model` accepte un id HuggingFace (téléchargé dans le cache HF local) ou un
répertoire local contenant des safetensors + config.json.
"""
import argparse
import json
import os
import random
import sys
from dataclasses import asdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from attention_obfuscation import obfuscate_attention_layer
from ffn_obfuscation import obfuscate_ffn_layer
from model_transform import ObfuscationKeys

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

_ATTR_TENSORS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight")
_BIAS_TENSORS = ("self_attn.q_proj.bias", "self_attn.k_proj.bias",
                 "self_attn.v_proj.bias")
_PASSTHROUGH_SUFFIXES = ("input_layernorm.weight",
                         "post_attention_layernorm.weight",
                         "self_attn.q_norm.weight", "self_attn.k_norm.weight")

_TOKEN_ID_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id",
                    "decoder_start_token_id")
_Q_NORM_TYPES = {"qwen3", "qwen3_moe"}


def _remap_dict_token_ids(holder, permutation):
    """Version dict de `model_transform._remap_token_ids` (les fichiers
    config/generation_config sont des dicts ici, pas des objets HF)."""
    if holder is None:
        return
    for field in _TOKEN_ID_FIELDS:
        value = holder.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            holder[field] = [permutation[int(v)] for v in value]
        else:
            holder[field] = permutation[int(value)]


def _vocab_permutation(seed, vocab_size):
    """Même tirage de permutation que `embedding_obfuscation.obfuscate_embedding`."""
    rng_py = random.Random(seed)
    permuted_ids = list(range(vocab_size))
    rng_py.shuffle(permuted_ids)
    permutation = dict(zip(range(vocab_size), permuted_ids))
    unpermute = {v: k for k, v in permutation.items()}
    perm_index = torch.tensor([permutation[i] for i in range(vocab_size)])
    return permutation, unpermute, perm_index


def _obfuscate_table(src_bf16, alpha, noise_seed, perm_index, chunk_rows):
    """Bruit + permutation d'une table (embed ou head), par blocs de lignes
    claires. Équivalent bit-à-bit de `obfuscate_embedding` :
    dst[Π(c)] = src[c] + α·σ(src)·noise[c], où noise suit le même flux
    `randn` que le tirage complet."""
    vocab_size, d = src_bf16.shape
    src = src_bf16.float()  # (vocab, d) float32 — seule grosse copie
    sigma = src.std()
    dst = torch.empty_like(src_bf16)
    out_dtype = dst.dtype
    gen = torch.Generator().manual_seed(noise_seed)
    for c0 in range(0, vocab_size, chunk_rows):
        c1 = min(c0 + chunk_rows, vocab_size)
        noise = torch.randn((c1 - c0, d), generator=gen)
        noisy = src[c0:c1] + alpha * sigma * noise
        dst[perm_index[c0:c1]] = noisy.to(out_dtype)
    return dst, sigma


def obfuscate_embedding_chunked(w_embed, w_head, alpha_e, alpha_h, seed,
                                chunk_rows=8192):
    """Équivalent bit-à-bit de `obfuscate_embedding(..., apply_key_matrices=False)`
    avec une empreinte mémoire réduite (par blocs de lignes).

    `w_embed`/`w_head` : tenseurs bf16 complets (vocab, d). Retourne les tables
    obfusquées en bf16 + la permutation/unpermutation du vocabulaire."""
    assert w_embed.shape == w_head.shape
    permutation, unpermute, perm_index = _vocab_permutation(
        seed, w_embed.shape[0])
    w_embed_obf, sigma_e = _obfuscate_table(
        w_embed, alpha_e, seed, perm_index, chunk_rows)
    w_head_obf, sigma_h = _obfuscate_table(
        w_head, alpha_h, seed + 1, perm_index, chunk_rows)
    return w_embed_obf, w_head_obf, permutation, unpermute, sigma_e, sigma_h


class _ConfigLike:
    """Mini-objet à attributs pour `obfuscate_layer_tensors`."""

    def __init__(self, config_dict):
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.num_key_value_heads = config_dict["num_key_value_heads"]
        self.hidden_size = config_dict["hidden_size"]
        self.head_dim = config_dict.get("head_dim")


def obfuscate_layer_tensors(config, i, tensors, seed, beta=8, gamma=1e3,
                            zeta=1e3, rope_scaling=True):
    """Obfusque les poids d'une couche (dict nom -> tenseur bf16).

    Gère les biais q/k/v s'ils sont présents (Qwen2) ou absents (Qwen3).
    Retourne le dict des tenseurs obfusqués en bf16 (sans les normes, qui
    passent telles quelles côté appelant)."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    d_head = config.head_dim or config.hidden_size // num_heads

    attn = f"model.layers.{i}.self_attn"
    mlp = f"model.layers.{i}.mlp"

    def _bias(name):
        if name in tensors:
            return tensors[name].float()
        return None

    obf_attn = obfuscate_attention_layer(
        tensors[f"{attn}.q_proj.weight"].float(),
        tensors[f"{attn}.k_proj.weight"].float(),
        tensors[f"{attn}.v_proj.weight"].float(),
        tensors[f"{attn}.o_proj.weight"].float(),
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
        beta=beta, gamma=gamma, zeta=zeta, seed=seed * 10000 + 2 * i,
        b_q=_bias(f"{attn}.q_proj.bias"),
        b_k=_bias(f"{attn}.k_proj.bias"),
        b_v=_bias(f"{attn}.v_proj.bias"),
        rope_layout="half", rope_scaling=rope_scaling,
    )
    obf_ffn = obfuscate_ffn_layer(
        tensors[f"{mlp}.gate_proj.weight"].float(),
        tensors[f"{mlp}.up_proj.weight"].float(),
        tensors[f"{mlp}.down_proj.weight"].float(),
        seed=seed * 10000 + 2 * i + 1,
    )

    out = {
        f"{attn}.q_proj.weight": obf_attn.w_q_obf.to(torch.bfloat16),
        f"{attn}.k_proj.weight": obf_attn.w_k_obf.to(torch.bfloat16),
        f"{attn}.v_proj.weight": obf_attn.w_v_obf.to(torch.bfloat16),
        f"{attn}.o_proj.weight": obf_attn.w_o_obf.to(torch.bfloat16),
        f"{mlp}.gate_proj.weight": obf_ffn.gate_proj_obf.to(torch.bfloat16),
        f"{mlp}.up_proj.weight": obf_ffn.up_proj_obf.to(torch.bfloat16),
        f"{mlp}.down_proj.weight": obf_ffn.down_proj_obf.to(torch.bfloat16),
    }
    if obf_attn.b_q_obf is not None:
        out[f"{attn}.q_proj.bias"] = obf_attn.b_q_obf.to(torch.bfloat16)
    if obf_attn.b_k_obf is not None:
        out[f"{attn}.k_proj.bias"] = obf_attn.b_k_obf.to(torch.bfloat16)
    if obf_attn.b_v_obf is not None:
        out[f"{attn}.v_proj.bias"] = obf_attn.b_v_obf.to(torch.bfloat16)
    return out


class _ShardWriter:
    """Accumule les tenseurs et les écrit en shards safetensors de taille
    bornée. Les tenseurs peuvent se répartir librement entre shards (le
    découpage n'a aucune contrainte de couche : l'index.json fait le lien)."""

    def __init__(self, out_dir, target_bytes=1.5e9):
        self.out_dir = out_dir
        self.target_bytes = target_bytes
        self.current = {}
        self.current_bytes = 0
        self.total_bytes = 0
        self.index = 0
        self.weight_map = {}

    def add(self, tensors):
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        if self.current and self.current_bytes + size > self.target_bytes:
            self._flush()
        self.current.update(tensors)
        self.current_bytes += size
        self.total_bytes += size

    def _flush(self):
        self.index += 1
        name = f"model-{self.index:05d}-of-NNNNN.safetensors"
        save_file(self.current, os.path.join(self.out_dir, name))
        for tensor_name in self.current:
            self.weight_map[tensor_name] = name
        self.current = {}
        self.current_bytes = 0

    def finish(self):
        if self.current:
            self._flush()
        total = self.index
        for i in range(1, total + 1):
            old = os.path.join(self.out_dir,
                               f"model-{i:05d}-of-NNNNN.safetensors")
            new = os.path.join(self.out_dir,
                               f"model-{i:05d}-of-{total:05d}.safetensors")
            os.rename(old, new)
        for name in self.weight_map:
            parts = self.weight_map[name].split("-")  # model-00001-of-NNNNN.*
            self.weight_map[name] = (f"model-{parts[1]}-of-{total:05d}"
                                     ".safetensors")
        return total


def _read_tensor(source_dir, shard, name):
    """Lit un seul tenseur d'un shard sans charger tout le fichier."""
    with safe_open(os.path.join(source_dir, shard), framework="pt",
                   device="cpu") as f:
        return f.get_tensor(name)


def transform_streaming(model_name, output_dir, seed, alpha_e=1.0, alpha_h=0.2,
                        lam=0.3, beta=8, gamma=1e3, zeta=1e3,
                        keys_path="obfuscation_keys.json", rope_scaling="auto",
                        chunk_rows=8192, shard_target_bytes=1.5e9):
    source_dir, _ = _resolve_source(model_name)

    with open(os.path.join(source_dir, "config.json")) as f:
        config_dict = json.load(f)
    config_type = config_dict.get("model_type")
    # bool strict — une chaîne "off"/"on" serait truthy et réactiverait Ĥ.
    if rope_scaling == "on":
        use_rope_scaling = True
    elif rope_scaling == "off":
        use_rope_scaling = False
    else:
        use_rope_scaling = config_type not in _Q_NORM_TYPES
    print(f"[streaming] model_type={config_type}, rope_scaling="
          f"{'on' if use_rope_scaling else 'off'}")

    num_heads = config_dict["num_attention_heads"]
    num_kv_heads = config_dict["num_key_value_heads"]
    assert num_heads % num_kv_heads == 0
    num_layers = config_dict["num_hidden_layers"]

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        # layout mono-fichier (petit modèle local, pas d'index) : on énumère
        # les clés du fichier unique.
        single = "model.safetensors"
        with safe_open(os.path.join(source_dir, single), framework="pt") as f:
            weight_map = {k: single for k in f.keys()}

    # --- embedding / unembedding (chunké, séquentiel : une table à la fois
    # pour borner la RAM : src float32 2,5 Go + sortie bf16 1,2 Go) ---
    permutation, unpermute, perm_index = _vocab_permutation(
        seed, config_dict["vocab_size"])
    embed_t = _read_tensor(source_dir, weight_map["model.embed_tokens.weight"],
                           "model.embed_tokens.weight")
    w_embed_obf, sigma_e = _obfuscate_table(
        embed_t, alpha_e, seed, perm_index, chunk_rows)
    del embed_t
    head_t = _read_tensor(source_dir, weight_map["lm_head.weight"],
                          "lm_head.weight")
    w_head_obf, sigma_h = _obfuscate_table(
        head_t, alpha_h, seed + 1, perm_index, chunk_rows)
    del head_t
    print(f"[streaming] σ(embed)={float(sigma_e):.4f}, "
          f"σ(head)={float(sigma_h):.4f} "
          f"(alpha_e={alpha_e}, alpha_h={alpha_h})")

    # --- couches : shard par shard, chaque couche complète est traitée puis
    # libérée ; les couches à cheval sur deux shards attendent dans `pending`
    # (une seule au plus, ~400 Mo). ---
    os.makedirs(output_dir, exist_ok=True)
    writer = _ShardWriter(output_dir, target_bytes=shard_target_bytes)
    writer.add({"model.embed_tokens.weight": w_embed_obf})
    del w_embed_obf

    pending = {}

    def _process_complete_layers():
        progressed = True
        while progressed:
            progressed = False
            for i in range(num_layers):
                prefix = f"model.layers.{i}."
                if {prefix + s for s in _ATTR_TENSORS} <= set(pending):
                    layer_in = {n: pending.pop(n) for n in list(pending)
                                if n.startswith(prefix)}
                    out = obfuscate_layer_tensors(
                        _ConfigLike(config_dict), i, layer_in, seed,
                        beta=beta, gamma=gamma, zeta=zeta,
                        rope_scaling=use_rope_scaling)
                    writer.add(out)
                    progressed = True
                    break

    shards = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    for shard, names in shards.items():
        # lecture tenseur par tenseur (safe_open, mmap) plutôt que `load_file`
        # : jamais plus d'un tenseur de couche en RAM à la fois.
        with safe_open(os.path.join(source_dir, shard), framework="pt",
                       device="cpu") as f:
            for name in names:
                t = f.get_tensor(name)
                if name in ("model.embed_tokens.weight", "lm_head.weight"):
                    del t  # déjà traités
                    continue
                if name == "model.norm.weight" or name.endswith(
                        _PASSTHROUGH_SUFFIXES):
                    writer.add({name: t})  # normes : copie à l'identique
                    del t
                    continue
                pending[name] = t
        _process_complete_layers()

    assert not pending, f"tenseurs de couche non traités: {sorted(pending)}"

    writer.add({"lm_head.weight": w_head_obf})
    del w_head_obf
    total = writer.finish()

    # index.json — `total_size` : somme exacte des octets de tenseurs écrits.
    with open(os.path.join(output_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {"total_size": writer.total_bytes},
                   "weight_map": writer.weight_map}, f)

    # --- config / generation_config : IDs spéciaux dans l'espace permuté ---
    for fname in ("config.json", "generation_config.json"):
        src = os.path.join(source_dir, fname)
        if not os.path.exists(src):
            continue
        with open(src) as f:
            holder = json.load(f)
        _remap_dict_token_ids(holder, permutation)
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(holder, f, indent=2)

    keys = ObfuscationKeys(permutation, unpermute, seed)
    with open(keys_path, "w") as f:
        json.dump(asdict(keys), f)
    print(f"[streaming] {total} shards écrits dans {output_dir}")
    print(f"[streaming] clés écrites dans {keys_path} (secret client — à ne "
          f"JAMAIS copier sur le serveur)")
    return keys


def _resolve_source(model_name):
    """Résout l'id HF (téléchargement dans le cache) ou un répertoire local."""
    if os.path.isdir(model_name):
        return model_name, model_name
    if snapshot_download is None:  # pragma: no cover
        raise SystemExit("huggingface_hub requis pour télécharger un modèle")
    cache = snapshot_download(
        model_name,
        allow_patterns=["config.json", "generation_config.json",
                        "model.safetensors.index.json", "model-*.safetensors"],
    )
    return cache, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="id HuggingFace ou répertoire local")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-h", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--beta", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1e3)
    parser.add_argument("--zeta", type=float, default=1e3)
    parser.add_argument("--keys", default="obfuscation_keys.json")
    parser.add_argument("--rope-scaling", choices=["auto", "on", "off"],
                        default="auto",
                        help="'auto' : off pour qwen3/qwen3_moe (q_norm/"
                             "k_norm), on sinon")
    parser.add_argument("--chunk-rows", type=int, default=8192,
                        help="lignes d'embedding traitées par bloc (mémoire)")
    parser.add_argument("--shard-target-bytes", type=float, default=1.5e9)
    args = parser.parse_args()
    transform_streaming(
        args.model, args.output, args.seed,
        alpha_e=args.alpha_e, alpha_h=args.alpha_h, lam=args.lam,
        beta=args.beta, gamma=args.gamma, zeta=args.zeta,
        keys_path=args.keys, rope_scaling=args.rope_scaling,
        chunk_rows=args.chunk_rows,
        shard_target_bytes=args.shard_target_bytes,
    )
