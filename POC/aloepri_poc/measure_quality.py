"""
Compare la perplexité du modèle obfusqué vs baseline sur un jeu de prompts
de test fixe.

Le modèle obfusqué n'a pas de tokenizer propre (cf. `server.py`) : ses lignes
d'embedding sont réindexées par `vocab_permutation` (écrite dans
`obfuscation_keys.json` par `model_transform.py`). Évaluer sa perplexité
exige donc de tokenizer avec le tokenizer PUBLIC de Qwen (comme le ferait
`client_wrapper.ClientCodec`), puis de permuter les IDs avant de les passer
au modèle obfusqué — les lui donner en clair reviendrait à lire la mauvaise
ligne d'embedding pour chaque token, et mesurerait le bruit d'une mauvaise
permutation plutôt que le coût réel de l'obfuscation.

La perplexité reste comparable au clair : la cross-entropy est invariante par
un ré-étiquetage bijectif du vocabulaire appliqué de façon cohérente aux
logits ET aux labels, ce qu'est exactement `vocab_permutation` (cf.
`tests/test_model_transform.py::test_tiny_qwen2_logits_are_preserved...`).

ATTENTION SÉCURITÉ : contrairement à `server.py` (qui ne voit jamais
`obfuscation_keys.json`), ce script de MESURE en a besoin localement — donc
sur le Pod, le temps de la mesure. C'est une dérogation volontaire, limitée
au POC/benchmarking (le développeur contrôle le Pod loué de bout en bout),
pas la posture de `server.py` en service. Supprimer `obfuscation_keys.json`
du Pod après usage (cf. RUNBOOK.md).

Usage:
    python measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct \
        --obfuscated ./obfuscated_model --keys obfuscation_keys.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_PROMPTS = [
    "La capitale de la France est",
    "Le théorème de Pythagore énonce que",
    "En 1789, la Révolution française",
    "L'eau bout à une température de",
    "La photosynthèse est le processus par lequel",
    "Albert Einstein a formulé la théorie de",
    "Le plus grand océan du monde est",
    "La Terre tourne autour du Soleil en environ",
    "Le symbole chimique de l'or est",
    "La tour Eiffel a été construite en",
    "Un triangle équilatéral possède",
    "La vitesse de la lumière dans le vide est d'environ",
    "Le Sahara est le plus grand désert",
    "Napoléon Bonaparte est devenu empereur en",
    "La monnaie officielle du Japon est",
    "Le fleuve le plus long d'Afrique est",
    "La Joconde a été peinte par",
    "Un octogone possède",
    "La capitale de l'Italie est",
    "Le corps humain contient environ",
    "La Seconde Guerre mondiale s'est terminée en",
    "Le plus haut sommet du monde est",
    "L'ADN est une molécule qui",
    "La devise de la République française est",
]


def permute_ids(ids, permutation):
    """Applique la permutation de vocabulaire à une liste d'IDs clairs.

    Fonction pure (pas de modèle, pas de GPU) — testable isolément."""
    if permutation is None:
        return list(ids)
    return [permutation[i] for i in ids]


def perplexity(model, tokenizer, text, permutation=None):
    ids = permute_ids(tokenizer.encode(text), permutation)
    ids = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return torch.exp(loss).item()


def main(baseline_path, obfuscated_path, keys_path):
    with open(keys_path) as f:
        keys = json.load(f)
    permutation = {int(k): int(v) for k, v in keys["vocab_permutation"].items()}
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)

    # Un modèle 7B à la fois sur le GPU (~14 Go en bf16 chacun) : les deux
    # simultanément (~28 Go) dépasseraient les 24 Go d'une RTX A5000, le GPU
    # visé par RUNBOOK.md. Séquentiel plutôt que simultané.
    baseline = AutoModelForCausalLM.from_pretrained(baseline_path, dtype=torch.bfloat16).cuda()
    baseline_ppl = [perplexity(baseline, tokenizer, p) for p in TEST_PROMPTS]
    del baseline
    torch.cuda.empty_cache()

    obfuscated = AutoModelForCausalLM.from_pretrained(obfuscated_path, dtype=torch.bfloat16).cuda()
    obfuscated_ppl = [perplexity(obfuscated, tokenizer, p, permutation) for p in TEST_PROMPTS]
    del obfuscated
    torch.cuda.empty_cache()

    deltas = []
    for prompt, b, o in zip(TEST_PROMPTS, baseline_ppl, obfuscated_ppl):
        delta_pct = (o - b) / b * 100
        deltas.append(delta_pct)
        print(f"{prompt[:40]:40s} baseline={b:.2f} obfusqué={o:.2f} delta={delta_pct:+.1f}%")

    print(f"\nmoyenne sur {len(TEST_PROMPTS)} prompts : delta={sum(deltas) / len(deltas):+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--obfuscated", required=True)
    parser.add_argument("--keys", default="obfuscation_keys.json",
                        help="clés produites par model_transform.py (nécessaires "
                             "ici uniquement, jamais par server.py — cf. docstring)")
    args = parser.parse_args()
    main(args.baseline, args.obfuscated, args.keys)
