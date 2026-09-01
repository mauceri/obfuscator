# État du projet — obfuscator (AloePri pour Qwen3-8B)

*Synthèse durable — mise à jour 2026-08-27. Les détails sont dans les rapports `artifacts/*.md` et le notebook (marqueurs `cellule N`).*

## Ce que c'est
Implémentation d'**AloePri** (arXiv 2603.01499) : inférence LLM confidentielle par
obfuscation covariante (transformations de données ET de poids), pour
**Qwen3-8B**, servie sur **Modal**. Machine `sanroque`, repo
`github.com/mauceri/obfuscator`. Venv : `~/Secretarius/Wiki_LM/.venv` ;
CLI Modal : `~/modal-venv/bin/modal`.

## Ce qui est fait
1. **POC h=0** (permutation + bruit α_e, attention obfusquée, FFN) :
   `aloepri/model_transform.py`, `transform_streaming.py` — servi et validé.
2. **Schéma complet h>0** (matrices clés P̂/Q̂ globales, d→d+2h, normes κ
   §5.2.5) : `aloepri/chained_transform.py` — validé 0.6B local + 8B Modal.
   Deux corrections de l'Algorithme 1 du papier : F1/F2 ~ N(0,1/d) ; scaling
   C √(h/d) (sans quoi ‖xP̂‖≈1.45 et le round-trip diverge).
3. **Attaques** : ISA (hidden/attn) `isa_attack.py` ; clean-space
   `attention_inversion.py` ; **VMA directe** `vma_attack.py` ; **VMA produit
   (Table 9)** `vma_product.py` + `vma_product_full` (toutes couches).
4. **Fine-tuning** : `modal_app.py::finetune_corpus` (corpus GEPA synthétique,
   embarqué dans l'image Modal).
5. **Notebook** pédagogique (50 cellules, marqueurs `cellule N`) — mi-papier,
   mi-cellules, structuré sur la Figure 2.
6. **Tests** : 58 (14 fichiers) — `pytest aloepri/tests/` (venv Wiki_LM).

## Résultats clés (mesures, avec références)
| Sujet | Résultat | Réf. |
|---|---|---|
| VMA directe (h=0) | Π récupérée ~99-100 % (même à α_e=1.0) → nécessite h>0 | `vma_report.md` |
| VMA produit 0.6B chaîné (8 couches, α=0.3) | 18,4 % ; à α=0 : 99,6 % | `vma_produit_8b_complet.md` |
| **VMA produit 8B h>0 (36 couches, 2 vues, vote)** | **0,0 %** | idem |
| Fine-tuning + bruit (0.6B, α=0.3) | 2,0 % (le FT est un filet, pas une défense seule) | idem |
| ISA hidden couche 18 (8B h>0) | **0 %** | `chained_8b_report.md` |
| ISA canal attn 0.6B | sous-déterminé : 0 % même baseline (canal hidden : 88,9 %) | `isa_report.md` |
| Qualité 8B h>0 (α=0.3) | capitale→Paris ; corr logits 0.94-0.975 ; top1 0.625 | `chained_8b_report.md` |
| Table 3 du papier (Qwen3-14B : VMA 25 % à α_e=1.0) | **non reproductible** avec notre procédure (réserve : leur métrique/vue inconnue) | `vma_produit_8b_complet.md` |

**Lecture sécurité (8B h>0, α_e=0.3)** : VMA directe impossible (d+2h ≠ d) ;
VMA produit 0 % ; ISA hidden 0 %. Résidu : vues V×V non testées + écart de
procédure avec le papier.

## Déploiement
- Service : `https://mauceri--obfuscator-aloepri-serve.modal.run` — health 200
  avec la clé (`~/.aloepri-api-key`), fail-closed (401 sans Bearer).
- Volumes Modal : `obfuscator-models` (qwen3-8b-obf [h=0, servi],
  qwen3-8b-obf-h128 [h>0], qwen3-06b-ft-gepa, variantes 0.6B) ;
  `obfuscator-keys` **supprimé** (posture : clés côté client uniquement).
- Clés locales : `artifacts/obfuscation_keys.json` (seed 0 — valides pour
  0.6B et 8B, même permutation).

## Pièges connus (importants)
- **`RUN_HEAVY=True` est laissé dans le notebook** (état de travail) : ne
  JAMAIS lancer `nbconvert --execute` headless — il lance les runs Modal réels
  (incident du 2026-08-27 : transform 8B déclenché par erreur).
- **Consigne globale** (`~/.dsh/AGENTS.md` règle 6) : aucune tâche longue
  (fine-tuning, runs Modal, calculs de plusieurs minutes) sans accord
  préalable avec budget.
- `verify()` : tolérance bf16 `_BF16_TOL=2e-3` (le strict bit-à-bit échouait
  sur 1-2 ulp bf16) ; les clés ne se re-téléchargent pas (cellule 2.3
  idempotente).
- Le notebook doit **calculer** ses chiffres — plus aucune valeur mesurée en
  dur dans les cellules (commit `495bf8b`).

## Ouvert / à faire
- Vues **V×V** de la Table 9 (gram q·k, W_e·W_h — 46 Go) : non testées
  (nécessite A100-80GB/H100 ou streaming disque).
- Benchmark de précision sur corpus (MMLU-like) du modèle 8B h>0.
- Canal **attention** ISA sur le modèle h>0 (seul le canal hidden a été
  re-mesuré).
- **IMA** (attaque par entraînement) : non implémentée (évidence papier 0 %).
- **Rotation de la permutation** (attaques par fréquence TFMA/SDA) : procédure
  opérationnelle à définir (client-side, sans re-transform).
- Extraction du grand corpus GEPA (~138 k textes) depuis
  `~/gen_corpus_gepa_codex/gepa_llm_calls.log` si besoin d'un corpus plus
  conséquent.
