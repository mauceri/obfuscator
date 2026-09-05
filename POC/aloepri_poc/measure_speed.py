"""
Compare tokens/s et latence entre modèle obfusqué et baseline.

Même remarque de permutation/sécurité que `measure_quality.py` : le modèle
obfusqué doit recevoir des IDs PERMUTÉS (via `vocab_permutation`, lue dans
`obfuscation_keys.json`), sinon `generate()` tourne sur des lignes
d'embedding qui ne correspondent à aucun token cohérent dans cet espace — un
token d'arrêt (EOS) mal placé côté permuté pourrait interrompre la génération
prématurément, faussant le nombre de tokens produits et donc le tok/s mesuré.
Ce script a donc besoin de `obfuscation_keys.json` localement (sur le Pod, le
temps de la mesure) — dérogation documentée dans `measure_quality.py` et
`RUNBOOK.md`, jamais nécessaire pour `server.py` en service.

Usage:
    python measure_speed.py --baseline Qwen/Qwen2.5-7B-Instruct \
        --obfuscated ./obfuscated_model --keys obfuscation_keys.json
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Décris en trois phrases le fonctionnement d'un transformer."


def tokens_per_second(n_generated, elapsed):
    """Fonction pure — testable isolément, sans modèle ni GPU."""
    return n_generated / elapsed


def measure(model, ids, max_new_tokens=100):
    ids = ids.to(model.device)
    start = time.perf_counter()
    output = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start
    n_generated = output.shape[1] - ids.shape[1]
    return tokens_per_second(n_generated, elapsed), elapsed


def main(baseline_path, obfuscated_path, keys_path):
    with open(keys_path) as f:
        keys = json.load(f)
    permutation = {int(k): int(v) for k, v in keys["vocab_permutation"].items()}
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    clear_ids = tokenizer.encode(PROMPT)
    permuted_ids = [permutation[i] for i in clear_ids]

    # Un modèle 7B à la fois sur le GPU (~14 Go en bf16 chacun) : les deux
    # simultanément (~28 Go) dépasseraient les 24 Go d'une RTX A5000, le GPU
    # visé par RUNBOOK.md. Séquentiel plutôt que simultané.
    baseline = AutoModelForCausalLM.from_pretrained(baseline_path, dtype=torch.bfloat16).cuda()
    b_tps, b_lat = measure(baseline, torch.tensor([clear_ids]))
    del baseline
    torch.cuda.empty_cache()

    obfuscated = AutoModelForCausalLM.from_pretrained(obfuscated_path, dtype=torch.bfloat16).cuda()
    o_tps, o_lat = measure(obfuscated, torch.tensor([permuted_ids]))
    del obfuscated
    torch.cuda.empty_cache()

    print(f"baseline : {b_tps:.1f} tok/s, {b_lat:.2f}s")
    print(f"obfusqué : {o_tps:.1f} tok/s, {o_lat:.2f}s")
    print(f"overhead vitesse : {(b_tps - o_tps) / b_tps * 100:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--obfuscated", required=True)
    parser.add_argument("--keys", default="obfuscation_keys.json",
                        help="clés produites par model_transform.py (nécessaires "
                             "ici uniquement, jamais par server.py — cf. docstring)")
    args = parser.parse_args()
    main(args.baseline, args.obfuscated, args.keys)
