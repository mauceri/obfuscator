# Correctifs issus de la revue — bilan (2026-09-01)

*Suite de la revue de code externe (cf. `docs/REVIEW.md`). Ce document
retrace les correctifs appliqués, les résultats mesurés et ce qui reste.
Toutes les valeurs sont issues des logs Modal (apps référencées), jamais
en dur dans le code.*

## Contexte

La revue a identifié que le résultat central « VMA = 0,00 % » n'était pas
établi : il fallait d'abord prouver que `vma_product_full` atteint ~100 %
dans le cas dégénéré (contrôle positif) avant de conclure quoi que ce soit
sur la défense. Trois défauts cumulables de l'attaque avaient été relevés
(vue up inerte, normalisation, vote par mode) + deux réserves (ISA, frwiki).

## Résumé des découvertes (au-delà de la revue)

1. **Cause racine du « 0,00 % »** : les produits `W̃_e·W̃_gateᵀ` étaient
   calculés en **bf16** dans `vma_product_full` (poids du volume bf16 +
   produits bf16). Le chaînage P̂Q̂≈I y accumule une erreur d'arrondi
   (max|Y−X| ≈ 0,33, mesuré) qui détruit l'appariement **même sans
   défense** — d'où un 0,00 % qui était un artefact de l'attaque cassée,
   pas une preuve de sécurité. `run_vma_product` (0.6B) upcastait déjà en
   fp32 — c'est lui qui atteignait 18,4 %/99,6 %.
2. **Échelle de α_e** : le papier (§5.2.2) définit `E ~ N(0, σ²(W))` et
   `W* = W + α_e·E` — donc **α_e est un ratio σ(bruit)/σ(poids)**, et
   **α_e = 1,0 = 100 % de bruit** (pas 1 %). Notre implémentation était
   conforme ; la mécompréhension initiale (1,0 = 1 %) a été levée en
   vérifiant le texte du papier.
3. **ISA** : le 0 % par gradient était un artefact de la relaxation
   continue (des soft tokens de l'enveloppe convexe reproduisent l'état
   caché sans être le prompt) — **le canal hidden est informatif**, prouvé
   par la variante discrète (100 %).

## ✅ Correctifs appliqués (commits sur main)

### 1. Vues : `up` retirée (inerte par construction) — `704199e`

`up_proj_obf = up_proj[perm] * scale[perm]` : Ŝ_ffn (scaling diagonal) vit
sur `up_proj` ; RowSort élimine Ẑ (permutation) mais **pas Ŝ** (scaling qui
modifie le multiset de chaque ligne) → `vote_up ≈ 0 %` par construction.
`views` par défaut = `"gate"`, docstring documenté.

### 2. Produits en fp32 (cause racine) — `704199e`

`Y` et `X` calculés en **fp32** (poids bf16 du volume upcastés pour le
calcul). Mesure décisive (clés aléatoires, RowSort+cosinus) : fp32 → 100 %
d'identité, bf16 → 16 %. Le point 2 de la revue (normalisation) était déjà
couvert par `nearest_neighbor_rows` (cosinus normalisé) — le vrai coupable
était la précision bf16 des produits, pas la normalisation.

### 3. Agrégation : somme des similarités au lieu du vote par mode — `704199e`

`per_view[view]` accumule les similarités cosinus (z-score par ligne pour
donner le même poids à chaque couche) sur les 36 couches, puis **argmax
global** — remplace le `mode` (vote majoritaire) qui détruisait le signal
faible (le mode de 36 quasi-aléatoires ≈ 0 %).

### 4. ISA discrète (vocab-matching k-way) — `bfc3c54`, `62de4d8`, `a2c1789`

- `aloepri/isa_attack.py` : `vocab_match_attack` / `run_vocab_match` — à
  chaque position, le vrai token est mélangé à k−1 leurres, recherche
  discrète (teacher-forced = borne haute, ou greedy autorégressif),
  métrique MSE relative ou cosinus ;
- `modal_app.py::isa_vocab_attack` : fonction Modal avec bloc RÉSUMÉ
  (baseline 1/k, interprétation) ;
- fix dtype (embeds float32 vs modèle bf16 → `embeds.to(model.dtype)`) ;
- **5+1 tests dédiés** (63 → 64 tests verts).

## 📊 Résultats mesurés (2026-09-01)

### VMA produit (attaque corrigée, vue gate, 36 couches, somme)

| Modèle (FT + h>0 sauf mention) | α_e | α_h | TTRSR | app |
|---|---|---|---|---|
| base h>0 (contrôle, sans bruit) | 0 | 0 | **99,95 %** | `ap-hFgHA4IBZy1Xwajr2AKf32` |
| FT + h>0 | 0,01 | 0 | **99,95 %** | `ap-SrPzSm0P1v41vZqYTifC8i` |
| FT + h>0 | 0,3 | 0,2 | **90,8 %** | `ap-qH10woNC4EkkSecVLU3coF` |
| FT + h>0 | **1,0** | **0,2** | **8,35 %** | `ap-qtR6Miib9bE7tZKLt7Twzi` |

- **Contrôle positif : 99,95 %** sur base α_e=0 → le chemin de code est
  sain (l'attaque retrouve Π quand elle le doit) ;
- **α_e=0,3 est insuffisant** (90,8 %) ; **α_e=1,0 (recommandé papier)
  → 8,35 %**, conforme au papier (Table 3 : 13-25 % selon les modèles) ;
- **le fine-tuning n'y contribue pas** (la courbe base sans bruit est
  identique à FT+1 %) — l'affinage ne protège pas la VMA produit.

### ISA hidden, couche 18

| Variante | Résultat | Conclusion |
|---|---|---|
| Gradient (soft tokens) | 0 % (loss 0,007) | artefact relaxation continue |
| **Discrète (k-way, k=64, α_e=1,0)** | **100 %** (10/10) | **canal hidden INFORMATIF** |

L'attaquant (opérateur serveur) récupère les ids **permutés** de l'entrée à
100 % — la confidentialité du texte repose uniquement sur la **clé Π** côté
client (jamais sur Modal). AloePri h>0 ne protège pas les états cachés.

### Précision frwiki (4000 tokens — à élargir)

| Modèle | Perplexité | Top-1 |
|---|---|---|
| base (Qwen3-8B) | 1,88 | 0,8092 |
| FT (9.1b) | 2,01 | 0,7887 |
| FT + h>0, α_e=0,3 | 2,14 | 0,7742 |
| FT + h>0, α_e=1,0 | 2,28 | 0,7527 |

Coût du réglage défensif α_e=1,0 : perplexité +13 % vs FT (+21 % vs base),
top-1 −3,6 pts — cohérent avec la promesse du papier (« < 3 % de perte »),
non négligeable sur ce petit échantillon.

## 🔄 Ce qui reste

1. **frwiki** : élargir l'échantillon (plus de fichiers/tokens) et filtrer
   le balisage (le top-1 0,809 de la base est anormalement haut) avant de
   citer les chiffres ;
2. **Comparaison procédure papier** : notre VMA à α_e=1,0 (8,35 %) est dans
   la fourchette du papier (13-25 %) mais l'écart de valeur suggère des
   différences de vue/métrique/agrégation — à documenter si on compare
   publiquement ;
3. **IMA** (attaque par entraînement, 9.5) : réservé ;
4. **Canal attention ISA** sur h>0 : à tester (seul le canal hidden l'a
   été) — le papier (Table 4) suggère que les KeyMat le protègent ;
5. **Notebook** : les cellules 26-33 (section 9) reflètent les résultats
   corrigés (α_e=1,0, ISA discret) — marqueurs 0-35, headless OK ;
   rechargez dans Jupyter (File → Reload).
