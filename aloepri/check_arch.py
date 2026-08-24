"""Vérification pré-vol des hypothèses du POC AloePri pour un modèle cible.

À lancer AVANT toute transformation, sur le modèle exact qu'on veut obfusquer.
Ne télécharge que les fichiers de config (+ le tokenizer si `--tokenizer`),
jamais les poids : c'est rapide et gratuit.

Usage:
    python check_arch.py --model Qwen/Qwen3-8B
    python check_arch.py --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct

Ce que le script valide (les hypothèses que `model_transform.py`/`
transform_streaming.py` supposent silencieusement sinon) :

1. GQA : `num_key_value_heads` défini et `num_attention_heads % num_key_value_heads == 0`
   (le POC décline l'obfuscation d'attention en MHA comme en MLA).
2. `head_dim` pair (les paires RoPE de `rope_transform.py` exigent d_head % 2 == 0).
3. Biais q/k/v : le POC les transforme s'ils existent (`bias=True` en dur dans
   Qwen2, `config.attention_bias` dans Qwen3) et les ignore sinon.
4. Normes de tête q_norm/k_norm (Qwen3) : imposent `rope_scaling=off` — le
   facteur diagonal Ĥ du papier ne commute pas avec une RMSNorm de tête
   (cf. docstring d'`attention_obfuscation.py`). Le POC détecte ça tout seul,
   ce script le rend visible et vérifie que le flag CLI cohérent est choisi.
5. Layout RoPE : Qwen2/Qwen3 utilisent `rotate_half` (paires (i, i+d/2)) →
   `rope_layout="half"` obligatoire.
6. Weight tying (`tie_word_embeddings`) : les deux branches sont gérées, mais
   il vaut mieux le savoir (Qwen3-8B : non lié).
7. Cohérence vocabulaire : `config.vocab_size` == taille du vocabulaire du
   tokenizer (la permutation du POC couvre `vocab_size` lignes d'embedding ;
   un tokenizer plus grand produirait des IDs hors bornes).
"""
import argparse
import json
import sys

from transformers import AutoConfig, AutoTokenizer

# Connaissances d'architecture stables (vérifiées sur modeling_qwen2/qwen3 de
# transformers au moment de l'écriture) — `--load` permet de les confirmer sur
# un modèle réel si besoin.
_HAS_Q_NORM = {"qwen3", "qwen3_moe"}          # RMSNorm de tête avant RoPE
_HARDCODED_QKV_BIAS = {"qwen2", "qwen2_moe"}  # bias=True en dur dans modeling_qwen2
_ROPE_HALF = {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "llama", "llama3"}


def _fmt(ok):
    return "OK " if ok else "KO "


def check(model_name, tokenizer_name=None, load=False, rope_scaling="auto"):
    print(f"== Vérification des hypothèses AloePri pour {model_name} ==\n")
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type
    problems = []

    def report(label, ok, detail):
        print(f"  [{_fmt(ok)}] {label}: {detail}")
        if not ok:
            problems.append(label)
        return ok

    # 1. GQA
    has_kv = hasattr(config, "num_key_value_heads")
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv = getattr(config, "num_key_value_heads", None)
    gqa_ok = bool(has_kv and num_heads and num_kv and num_heads % num_kv == 0)
    report("GQA", gqa_ok,
           f"num_attention_heads={num_heads}, num_key_value_heads={num_kv}")

    # 2. head_dim
    d_head = getattr(config, "head_dim", None) or (
        config.hidden_size // num_heads if num_heads else None)
    report("head_dim pair (RoPE)", bool(d_head and d_head % 2 == 0),
           f"head_dim={d_head}, hidden_size={config.hidden_size}")

    # 3. Biais q/k/v
    if model_type in _HARDCODED_QKV_BIAS:
        bias = True
        bias_src = "bias=True en dur (modeling_qwen2)"
    elif hasattr(config, "attention_bias"):
        bias = config.attention_bias
        bias_src = f"config.attention_bias={config.attention_bias}"
    else:
        bias, bias_src = None, "non spécifié (détecté au chargement)"
    report("biais q/k/v", True, str(bias_src))

    # 4. q_norm/k_norm → rope_scaling
    has_qnorm = model_type in _HAS_Q_NORM
    if has_qnorm:
        detail = (f"{model_type} a q_norm/k_norm (RMSNorm de tête avant RoPE) "
                  "→ le facteur Ĥ du papier ne commute pas → rope_scaling=off "
                  "requis (le POC le choisit automatiquement, 'auto')")
        forced_ok = rope_scaling != "on"
        report("rope_scaling (q_norm/k_norm)", forced_ok,
               detail + ("" if forced_ok else
                         " MAIS --rope-scaling on forcé : round-trip cassé"))
    else:
        detail = (f"{model_type} sans q_norm/k_norm → rope_scaling=on OK "
                  "(comportement historique du POC)")
        forced_ok = rope_scaling != "off"
        report("rope_scaling (q_norm/k_norm)", forced_ok,
               detail + ("" if forced_ok else
                         " MAIS --rope-scaling off forcé inutilement"))

    # 5. Layout RoPE
    report("rope_layout", model_type in _ROPE_HALF,
           f"model_type={model_type} → rotate_half (paires (i, i+d/2)) "
           f"= rope_layout 'half' attendu")

    # 6. Weight tying
    tied = bool(getattr(config, "tie_word_embeddings", False))
    report("weight tying", True,
           f"tie_word_embeddings={tied} "
           + ("→ branche `tied` (embed==head)" if tied
              else "→ embed_tokens et lm_head transformés séparément"))

    # 7. Vocabulaire (padding autorisé : tokenizer ≤ embedding)
    tokenizer = None
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tv = len(tokenizer)
        if tv > config.vocab_size:
            report("vocabulaire", False,
                   f"tokenizer={tv} > config.vocab_size={config.vocab_size} : "
                   "des IDs produits par le tokenizer sortiraient de la table "
                   "d'embedding — bloquant")
        else:
            report("vocabulaire", True,
                   f"tokenizer={tv} ≤ config.vocab_size={config.vocab_size} "
                   f"({config.vocab_size - tv} lignes de padding inutilisées, "
                   "sans effet)")
    else:
        print("  [.. ] vocabulaire: non vérifié (passer --tokenizer pour "
              f"comparer ; config.vocab_size={config.vocab_size})")

    # 8. Modèle complet chargé (optionnel, coûteux)
    if load:
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype="bfloat16", low_cpu_mem_usage=True)
            attn0 = model.model.layers[0].self_attn
            has_qnorm_real = hasattr(attn0, "q_norm")
            b_q_real = attn0.q_proj.bias is not None
            report("chargé (q_norm réel)", has_qnorm_real == has_qnorm,
                   f"q_norm présent: {has_qnorm_real}, biais q: {b_q_real}")
            del model
        except Exception as e:  # pas bloquant : --load est une confirmation
            print(f"  [.. ] chargement complet impossible: {e}")

    print()
    if problems:
        print(f"RÉSULTAT : {len(problems)} hypothèse(s) en échec -> {problems}")
        print("Ne pas lancer la transformation avant de traiter ces points.")
        return 1
    print("RÉSULTAT : toutes les hypothèses sont satisfaites — la "
          "transformation peut être lancée.")
    print("Flags recommandés : --rope-scaling " + ("off" if has_qnorm else "on"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--load", action="store_true",
                        help="charger le modèle complet pour confirmer "
                             "q_norm/biais (coûteux, 16 Go sur Qwen3-8B)")
    parser.add_argument("--rope-scaling", choices=["auto", "on", "off"],
                        default="auto",
                        help="valeur CLI qu'on compte passer à la "
                             "transformation ; 'on' sur une architecture à "
                             "q_norm/k_norm est signalé comme erreur")
    args = parser.parse_args()
    sys.exit(check(args.model, tokenizer_name=args.tokenizer, load=args.load,
                   rope_scaling=args.rope_scaling))
