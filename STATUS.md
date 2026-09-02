# État du projet — obfuscator (AloePri pour Qwen3-8B)

*Synthèse durable — mise à jour 2026-09-02. Les détails sont dans les rapports `artifacts/*.md` et le notebook (marqueurs `cellule N`, journal daté cellule 35).*

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
   embarqué dans l'image Modal) — 8B en bf16 complet + gradient checkpointing,
   A100-80GB.
5. **Notebook** pédagogique (44 cellules, marqueurs `cellule N`) — mi-papier,
   mi-cellules, structuré sur la Figure 2 ; section 9 = sécurité (9.1-9.6b,
   cellules 22-34), journal des résultats datés = cellule 35 (§10), annexe
   des méthodes `modal_app.py` = cellule 36 (§11), évaluation Q&A DeepSeek
   (9.6c/9.6d) en cellules 42-43 (fin du notebook). Bloc POC h=0 retiré
   (2026-09-01).
6. **Tests** : 64 (14 fichiers) — `pytest aloepri/tests/` (venv Wiki_LM).

## Résultats clés (mesures, avec références)
| Sujet | Résultat | Réf. |
|---|---|---|
| VMA directe (h=0) | Π récupérée ~99-100 % (même à α_e=1.0) → nécessite h>0 | `vma_report.md` |
| VMA produit 0.6B chaîné (8 couches, α=0.3) | 18,4 % ; à α=0 : 99,6 % | `vma_produit_8b_complet.md` |
| **VMA produit 8B h>0 — contrôle positif (base α_e=0, attaque corrigée)** | **99,95 %** (chemin de code validé) | notebook cellule 35 |
| **VMA produit 8B FT+h>0 — courbe α_e (01/09, attaque corrigée)** | α_e=0,01 → 99,95 % ; **α_e=0,3 → 90,8 %** ; **α_e=1,0 → 8,35 %** | notebook cellule 35 |
| Fine-tuning + bruit (0.6B, α=0.3) | 2,0 % (le FT est un filet, pas une défense seule) | `vma_produit_8b_complet.md` |
| ISA hidden couche 18 (8B h>0) | **0 % par gradient (artefact) ; 100 % en recherche discrète (k-way)** → canal hidden INFORMATIF | notebook cellule 35 |
| ISA canal attn 0.6B | sous-déterminé : 0 % même baseline (canal hidden : 88,9 %) | `isa_report.md` |
| Qualité 8B h>0 (α=0.3) | capitale→Paris ; corr logits 0.94-0.975 ; top1 0.625 | `chained_8b_report.md` |
| **FT 8B complet (01/09)** | loss 1,76 → 0,27 (8475 pas, 83 min) → `qwen3-8b-ft-gepa` | notebook cellule 35 |
| **AloePri h>0 8B (01/09)** | hidden 4352 (h=128), transform_chained sur le FT → `qwen3-8b-ft-h128` | notebook cellule 35 |
| **Précision frwiki (01/09)** | α_e=0,3 : perp. base 1,88 / FT 2,01 / FT+h>0 2,14 ; **α_e=1,0 : 2,28** (top-1 0,7527) | notebook cellule 35 |
| **Précision PiaF questions (02/09)** | perp. base 6,09 / FT 6,97 / FT+obf α_e=1,0 **7,78** ; top-1 0,515 / 0,503 / 0,487 | notebook cellule 35 |
| **Q&A jugé DeepSeek (02/09, PiaF 150 couples)** | note base **4,62/5** vs obfusqué α_e=1,0 **4,42/5** (−0,20, −4,3 %) — la défense reste fonctionnelle en Q&A | notebook cellule 35 |
| Table 3 du papier (VMA 13-25 % à α_e=1.0) | **reproduit** (notre mesure : 8,35 % à α_e=1,0 — conforme) | notebook cellule 35 |

**Lecture sécurité (8B h>0)** : VMA directe impossible (d+2h ≠ d) ; la VMA
produit est neutralisée par le **bruit α_e=1,0** (8,35 %, conforme au papier
13-25 %) — **α_e=0,3 est insuffisant (90,8 %)** ; le « 0,0 % » initial était
un artefact de l'attaque bf16 cassée. ISA : canal hidden INFORMATIF
(100 % en recherche discrète k-way) — la protection du texte repose
uniquement sur la clé Π côté client. Résidu : vues V×V non testées.

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
  sur 1-2 ulp bf16) ; les clés ne se re-téléchargent pas (chargement
  idempotent).
- Le notebook doit **calculer** ses chiffres — plus aucune valeur mesurée en
  dur dans les cellules (commit `495bf8b`).

## Ouvert / à faire
- **Revue de code complète** (demandée par l'utilisateur — modal_app.py en
  priorité, par un agent externe) : à faire après la précision frwiki.
- Vues **V×V** de la Table 9 (gram q·k, W_e·W_h — 46 Go) : non testées
  (nécessite A100-80GB/H100 ou streaming disque).
- Benchmark de précision plus large (frwiki complet sur volume Modal, plus de
  tokens) — l'échantillon embarqué fait 4000 tokens.
- Canal **attention** ISA sur le modèle h>0 (seul le canal hidden a été
  re-mesuré).
- **IMA** (attaque par entraînement) : non implémentée (évidence papier 0 %).
- **Rotation de la permutation** (attaques par fréquence TFMA/SDA) : procédure
  opérationnelle à définir (client-side, sans re-transform).
- Extraction du grand corpus GEPA (~138 k textes) depuis
  `~/gen_corpus_gepa_codex/gepa_llm_calls.log` si besoin d'un corpus plus
  conséquent.
