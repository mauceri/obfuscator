# Réécriture de la section attaques du notebook — design (2026-08-27)

## Contexte et problème

Le notebook `notebooks/aloepri_procedure.ipynb` contient, à partir de la
cellule 36 (« 9. Attaques »), une section incohérente avec la stratégie du
projet :
- vestiges du **POC h=0** et du **Qwen3-0.6B** (variantes partielles, canal
  attn sous-déterminé) ;
- **valeurs mesurées en dur** dans les prints et le markdown (déjà nettoyées,
  commit `495bf8b`) ;
- **numérotation incohérente** (« 9. » avec sous-sections 5.x) ;
- le canal ISA utilisé (attn) ne montre pas la démonstration attendue.

Le projet se concentre sur **Qwen3-8B** et le **schéma complet AloePri h>0**.

## Stratégie cible (validée avec l'utilisateur)

La nouvelle section « 9. Sécurité — de l'entraînement à l'évaluation » suit
la stratégie en 6 étapes :

1. **Fine-tuning complet** de Qwen3-8B sur un corpus synthétique généré avec
   **DeepSeek + gen_corpus_gepa_codex** (GEPA/DSPy) ;
2. **AloePri complet** (h>0, matrices clés, α_e=0.3) sur le modèle fine-tuné ;
3. **VMA complète** (Table 9, toutes les couches, vote ; W_e·W_h écarté) ;
4. **ISA** (canal hidden) ;
5. **IMA** — plus tard (cellule réservée) ;
6. **Précision** sur `~/corpus_fr/frwiki` (perplexité + next-token top-1).

## Décisions de design (validées)

- **Génération du corpus** : la cellule markdown s'appuie sur le README de
  `gen_corpus_gepa_codex` (local `~/gen_corpus_gepa_codex/README.md` et
  GitHub) : GEPA (optimisation de prompt DSPy), `generate_corpus_dspy.py`
  avec `DEEPSEEK_API_KEY` comme générateur, règles de validation (7 clés
  JSON : `contenu`, `url`, `date`, `expressions_clefs`, `type_de_document`,
  `theme`, `categorie`), CLI commentée.
- **Fine-tuning** : **full** (tous les paramètres, pour modifier les poids —
  but de la stratégie), sur **A100-80GB** (mémoire ~70 Go : modèle 16 Go +
  gradients + AdamW). La cellule détaille `finetune_corpus` (déjà
  implémentée dans `modal_app.py`) : chargement bf16, tokenisation du corpus
  GEPA, boucle AdamW + `torch.amp` bf16, sauvegarde sur le volume.
- **Précision frwiki** : **perplexité + précision next-token top-1** sur un
  échantillon de `~/corpus_fr/frwiki` (1,1 Go), **comparaison** : base vs
  fine-tuné vs fine-tuné+obfusqué (h>0).

## Structure des nouvelles cellules (remplace les cellules 36-49)

| N° | Contenu | Type |
|---|---|---|
| 9.1a | Génération du corpus avec DeepSeek + GEPA (README, règles de validation, CLI) | markdown bref |
| 9.1b | Full fine-tuning Qwen3-8B — `finetune_corpus` détaillé, A100-80GB, budget annoté | code détaillé |
| 9.2 | AloePri complet (h>0, α_e=0.3) — `transform_chained` recopié/commenté (P̂/Q̂, κ, corrections Algo 1), sortie `qwen3-8b-ft-h128` | code détaillé |
| 9.3 | VMA complète — `vma_product_full` recopié/commenté : gate+up × 36 couches + vote ; W_e·W_h écarté (documenté) ; vues V×V = limite | code détaillé |
| 9.4 | ISA canal hidden (L1) — `isa_attack` recopié/commenté | code détaillé |
| 9.5 | IMA — réservé (« plus tard ») | markdown 1 ligne |
| 9.6 | Précision frwiki — perplexité + top-1 next-token, comparaison des 3 variantes | code détaillé |

## Contraintes (consignes utilisateur)

- **Zéro 0.6B / zéro h=0** dans la section (aucune mention du POC).
- Markdown **brefs** ; code Python **détaillé, recopié et commenté** pour les
  appels Modal (`!modal run modal_app.py::...`) — chaque option expliquée.
- **Aucune valeur mesurée en dur** : les taux viennent des cellules
  d'exécution ou des rapports `artifacts/*`.
- Marqueurs **`cellule N`** conservés (renumérotés selon la nouvelle
  structure).
- Les sections 1-8 restent **inchangées** pour l'instant.

## Budgets Modal (annotés dans les cellules, RUN_HEAVY=True)

FT 8B ~30-60 min A100-80GB (~4-8 $) ; transform h>0 ~1-1,5 h CPU (~1-2 $) ;
VMA ~1-1,5 h A100-40GB (~2-4 $) ; ISA ~30 min A100-40GB (~1-2 $) ;
précision ~20-40 min A100-40GB (~1-2 $). Total ~4-6 h / ~10-17 $ par run
complet.

## Validation

- JSON du notebook valide ; exécution headless **avec RUN_HEAVY forcé à
  False** (ne jamais exécuter tel quel : `RUN_HEAVY=True` lance les runs
  Modal réels — piège documenté).
- Vérification visuelle des cellules 9.1-9.6 (contenu, commentaires,
  numérotation).
