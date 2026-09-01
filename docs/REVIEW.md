# Revue de code — obfuscator (AloePri pour Qwen3-8B)

*Document de contexte pour une revue externe (Claude). Préparé le
2026-09-01. L'objectif : permettre à un reviewer qui ne connaît pas le
projet d'auditer `modal_app.py` (priorité) et le package `aloepri/` en
toute autonomie.*

## 1. Ce que fait le projet

Implémentation d'**AloePri** (arXiv 2603.01499) : inférence LLM
confidentielle par **obfuscation covariante** — on transforme les **poids**
du modèle ET les **données** (ids des tokens) de façon inversible, de sorte
que l'inférence sur le modèle obfusqué avec des ids permutés donne le même
résultat que l'inférence en clair, tout en empêchant l'opérateur du serveur
(qui possède les poids et observe les états internes) de reconstruire le
texte. Cible : **Qwen3-8B**, orchestré sur **Modal**.

Posture de sécurité : les **clés ne sont jamais sur Modal** (le volume
`obfuscator-keys` a été supprimé) ; elles vivent côté client
(`artifacts/obfuscation_keys.json`, seed 0). Le service est fail-closed
(Bearer token).

## 2. Structure du dépôt

```
modal_app.py            # fonctions Modal (14) : transform, attaques, FT, service
aloepri/                # package cœur
  key_matrix.py         # Algorithme 1 : P̂/Q̂, E1/E2, F1/F2, C
  chained_transform.py  # AloePri h>0 : reconstruction d+2h, chaînage P̂/Q̂, κ
  model_transform.py    # obfuscation h=0 (miniature + streaming)
  embedding_obfuscation.py / attention_obfuscation.py / ffn_obfuscation.py
  block_perm.py / rope_transform.py
  transform_streaming.py / verify_transform.py / check_arch.py
  vma_attack.py / vma_product.py / isa_attack.py / attention_inversion.py
  tests/                # 58 tests (14 fichiers)
notebooks/aloepri_procedure.ipynb  # procédure pédagogique, 36 cellules
artifacts/              # gitignoré : rapports de mesures + clés locales
docs/superpowers/       # specs et plans approuvés (contexte de conception)
STATUS.md               # synthèse durable, résultats mesurés
README.md               # prérequis + vérification headless
```

## 3. Contexte scientifique (pour juger le code)

Le papier définit des matrices clés $\hat P \in \mathbb{R}^{d\times(d+2h)}$
et inverses $\hat Q \in \mathbb{R}^{(d+2h)\times d}$ avec $\hat P\hat Q = I$
(Algorithme 1, §5.2.1), et un chaînage inter-couches (§5.4). Notre
implémentation comporte **deux corrections documentées de l'Algorithme 1**,
sans lesquelles le round-trip diverge :

1. **F1/F2 ~ N(0, 1/d)** (le papier suggérait 1/h) — `key_matrix.py` ;
2. **scaling des lignes de C par √(h/d)** — sans quoi ‖xP̂‖ ≈ 1.45 et le
   round-trip casse.

Conventions poids : **`(out, in)`** partout. Transformation unilatérale :
`embed·P̂` ; `head·Q̂ᵀ` ; `q/k/v/gate/up·(W·Wnorm)·Q̂ᵀ` ; `o/down·P̂ᵀ·W` ;
les RMSNorm de tête Qwen3 (`q_norm`/`k_norm`, γ appris non constant)
imposent `rope_rotation=False` (correctif 2026-08-24).

## 4. Ce qu'il faut revoir en priorité

### 4.1 `modal_app.py` (priorité absolue)
Les 14 fonctions Modal. Points sensibles connus :
- **`finetune_corpus`** : full FT 8B — bf16 complet (poids + grads + états
  AdamW via `zeros_like`), gradient checkpointing, A100-80GB. Vérifier la
  gestion mémoire, la sauvegarde sur volume, la reproductibilité (seed).
- **`transform_chained`** : reconstruction hidden 4352 + chaînage P̂/Q̂.
  Vérifier que `--model-name` accepte un chemin de volume (`/models/...`) et
  que la cible par défaut est cohérente (bug corrigé le 01/09 : il fallait
  obfusquer le modèle **fine-tuné**, pas la base).
- **`vma_product_full`** : attaque Table 9 — produits V×inter en bf16,
  `_row_sort_chunked` (le sort GPU bf16 upcast fp32 → OOM, d'où le
  chunking), appariement NN, votes. Vérifier la libération mémoire
  (`torch.cuda.empty_cache`) et l'exactitude des votes.
- **`precision_frwiki`** : lit un échantillon NDJSON embarqué
  (`/frwiki_sample`), régénère la permutation par seed (pas de lecture de
  clés dans le conteneur), calcule perplexité + top-1.
- **`serve`** : fail-closed, `enable_thinking=False`, blocage du token
  `<think>`. Vérifier la posture (aucune clé, aucun tokenizer).
- **`TRANSFORM_IMAGE`** : `add_local_*` avec `copy=True` partout (contrainte
  Modal : pas de build step après un mount).

### 4.2 `aloepri/`
- `key_matrix.py` : les deux corrections de l'Algorithme 1 (vérifier
  l'orthogonalité de Z, la nullité de C dans null(Fᵀ), la validité de
  P̂Q̂ = I en pratique).
- `chained_transform.py` : `estimate_kappa` / `calibrate_kappas` (κ §5.2.5,
  empirique par couche), ordre des opérations par couche.
- `vma_product.py` : `_row_sort_chunked` (correction OOM), le produit
  `W̃_e·W̃_gateᵀ` et l'élimination de Ẑ_ffn par RowSort.
- `isa_attack.py` : inversion par soft tokens + recuit de température.
- `verify_transform.py` : tolérance `_BF16_TOL = 2e-3` (le strict bit-à-bit
  échouait sur 1-2 ulp bf16) — vérifier que la tolérance ne masque pas une
  vraie erreur.

### 4.3 Tests
58 tests (14 fichiers), `pytest aloepri/tests/` (venv Wiki_LM). Vérifier :
couverture des corrections (F1/F2, scaling C), des conventions (out,in), du
round-trip chaîné, des attaques. Signaler tout test qui teste l'implémentation
plutôt que le contrat.

## 5. Points d'attention / questions ouvertes

1. **Reproductibilité** : seed partout ? Les permutations dépendent-elles de
   l'ordre des opérations Python/NumPy entre versions ?
2. **Déterminisme Modal** : `random.Random(seed)` vs `np.random` vs
   `torch.manual_seed` — sont-ils synchronisés dans chaque fonction ?
3. **Vues V×V de la Table 9** (gram q·k, W_e·W_h — 46 Go) : non testées,
   documenté comme limite A100-40GB. Est-ce acceptable, ou faut-il un
   streaming disque ?
4. **La tolérance bf16 de `verify`** (2e-3) : justifiée, mais vérifier
   qu'elle ne laisse pas passer des erreurs de transformation réelles.
5. **`compare_poc`** : fonction de référence interne — toujours utile ?
6. **Robustesse des chemins** : les fonctions Modal prennent des noms de
   sous-répertoires (`qwen3-8b-ft-h128`) ; vérifier la validation des entrées
   (chemins, params) et le comportement si un modèle est absent du volume.
7. **Coût/budgets** : les durées annotées (notebook) sont des estimations ;
   les mesurer précisément pour les mettre à jour.
8. **Sécurité** : la permutation est le secret ultime — vérifier qu'aucun
   chemin de code ne la fait fuiter (logs, erreurs, retours Modal).

## 6. Comment exécuter

```bash
# Tests (venv Wiki_LM)
source ~/Secretarius/Wiki_LM/.venv/bin/activate
python -m pytest aloepri/tests/ -q          # 58 tests

# Vérification headless du notebook (RUN_HEAVY=False — ne JAMAIS exécuter
# l'original avec RUN_HEAVY=True : il lance des runs Modal réels)
# (voir README.md — validation sur copie uniquement)

# Fonctions Modal : exécution via
#   ~/modal-venv/bin/modal run modal_app.py::<fonction> --<args>
# (nécessite une CLI Modal authentifiée et un compte avec quota GPU)
```

## 7. Livrables attendus de la revue

- Liste des bugs / risques classés par sévérité (critique / majeur / mineur) ;
- Vérification des deux corrections de l'Algorithme 1 et des conventions
  (out,in) ;
- Avis sur la gestion mémoire (finetune_corpus, vma_product_full) ;
- Avis sur la posture sécurité (aucune fuite de clé/permutation) ;
- Suggestions de tests manquants.
