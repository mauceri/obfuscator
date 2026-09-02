"""Rotation de niveau 1 (secours) : nouvelle permutation Π, SANS re-transform.

Réordonne les lignes de l'embedding et de la head d'un modèle obfusqué
existant avec une NOUVELLE permutation aléatoire (CSPRNG), et écrit la
nouvelle clé locale. Le reste du modèle (couches internes, P̂/Q̂, bruits)
est inchangé.

Mécanique : si l'ancien modèle indexe ses lignes par Π_old (ligne j =
données du token clair unperm_old[j]), alors pour une nouvelle clé Π_new il
suffit de copier, pour chaque ligne j, la ligne perm_old[unperm_new[j]] de
l'ancien modèle — sur `model.embed_tokens.weight` ET `lm_head.weight`.

Usage :
    .venv/bin/python tools/rotate_pi.py \\
        --model-in  qwen3-8b-ft-h128-a1-h02/ \\
        --model-out qwen3-8b-ft-h128-a1-h02-r1/ \\
        --keys-in   artifacts/obfuscation_keys.json \\
        --keys-out  artifacts/obfuscation_keys-r1.json

Puis :
  1. remplacer le modèle servi sur le volume Modal par `--model-out` ;
  2. mettre à jour le client avec `--keys-out` (artifacts/obfuscation_keys.json).

LIMITE (importante) : cette rotation protège contre la réutilisation d'une
clé compromise, mais PAS contre l'appariement différentiel multi-snapshots
(un serveur qui conserve l'ancien et le nouveau modèle peut relier leurs
lignes — les bruits α_e/α_h et les matrices P̂/Q̂ sont inchangés). Pour cela,
voir la rotation COMPLÈTE (docs/rotation-cles.md, niveau 2).
"""
import argparse
import json
import os
import random
import shutil

from safetensors.torch import load_file, save_file


def _new_permutation(vocab_size):
    """Permutation aléatoire cryptographique (pas random.Random seeded)."""
    rng = random.SystemRandom()
    permuted = list(range(vocab_size))
    rng.shuffle(permuted)
    perm = dict(zip(range(vocab_size), permuted))     # {clair: obfusqué}
    unperm = {v: k for k, v in perm.items()}
    return perm, unperm


def _load_keys(path):
    with open(path) as f:
        k = json.load(f)
    return ({int(a): int(b) for a, b in k["vocab_permutation"].items()},
            {int(a): int(b) for a, b in k["vocab_unpermute"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-in", required=True, help="modèle obfusqué (dossier safetensors)")
    ap.add_argument("--model-out", required=True, help="modèle réordonné (nouveau dossier)")
    ap.add_argument("--keys-in", default="artifacts/obfuscation_keys.json")
    ap.add_argument("--keys-out", required=True,
                    help="nouvelle clé (Π_new, Π_new⁻¹) — SECRET client")
    args = ap.parse_args()

    perm_old, _ = _load_keys(args.keys_in)
    V = len(perm_old)

    # nouvelle clé aléatoire (CSPRNG)
    perm_new, unperm_new = _new_permutation(V)
    # pour chaque ligne j du nouveau modèle : ligne à copier de l'ancien
    order = [perm_old[unperm_new[j]] for j in range(V)]

    os.makedirs(args.model_out, exist_ok=True)

    # copier les fichiers non-safetensors (config, tokenizer, etc.) tels quels
    for name in os.listdir(args.model_in):
        p = os.path.join(args.model_in, name)
        if os.path.isfile(p) and not name.endswith(".safetensors"):
            shutil.copy2(p, os.path.join(args.model_out, name))

    # réordonner embed + head dans chaque shard qui les contient
    n_tensors = 0
    for name in sorted(os.listdir(args.model_in)):
        if not name.endswith(".safetensors"):
            continue
        src = os.path.join(args.model_in, name)
        data = load_file(src)
        tensors = {}
        for tname, t in data.items():
            if tname == "model.embed_tokens.weight" or tname == "lm_head.weight":
                t = t[order]                     # (V, d2) réordonné
                n_tensors += 1
            tensors[tname] = t
        save_file(tensors, os.path.join(args.model_out, name))
        del data, tensors

    # nouvelle clé locale
    with open(args.keys_out, "w") as f:
        json.dump({"vocab_permutation": {str(k): v for k, v in perm_new.items()},
                   "vocab_unpermute": {str(k): v for k, v in unperm_new.items()},
                   "seed": None}, f, ensure_ascii=False)

    print(f"tenseurs réordonnés : {n_tensors} (embed + head, {V} lignes)")
    print(f"modèle  : {args.model_in} → {args.model_out}")
    print(f"clé     : {args.keys_in} → {args.keys_out} (seed None = aléatoire)")
    print("Rappel : remplacer le modèle servi sur le volume, puis mettre à "
          "jour le client avec la nouvelle clé. Rotation de SECOURS — voir "
          "docs/rotation-cles.md pour la rotation complète.")


if __name__ == "__main__":
    main()
