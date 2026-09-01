# Plan de correctifs — suite à la revue (2026-09-01)

*Conséquences de la revue externe. À appliquer APRÈS la fin du contrôle
positif (ne pas modifier le code pendant qu'un run Modal l'embarque).*

## Contexte

La revue a identifié que le résultat central « VMA = 0,00 % » n'est pas
établi tant qu'un **contrôle positif** n'a pas prouvé que `vma_product_full`
atteint ~100 % dans le cas dégénéré. Trois défauts cumulables de l'attaque,
plus deux réserves (ISA, frwiki).

## 0. Contrôle positif (EN COURS — run `ap-qNSTWLHldaGBuNiEZ3hW1n`)

- `transform_chained --alpha-e 0 --alpha-h 0 --model-name Qwen/Qwen3-8B
  --out-subdir qwen3-8b-base-h128-a0` (base NON affinée, zéro bruit) ;
- puis `vma_product_full --model-subdir qwen3-8b-base-h128-a0`.

**Interprétation attendue** :
- TTRSR ≈ 100 % → le chemin de code est sain, le 0 % (α_e=0,3 + FT) est la
  défense → mesurable et publiable ;
- TTRSR ≪ 100 % → l'attaque est cassée par un défaut d'implémentation ; le
  0 % ne prouve rien. Corriger les points 1-3 puis re-tester.

## 1. Vues : `up` inerte par construction

**Constat** (ffn_obfuscation.py) : `up_proj_obf = up_proj[perm] * scale[perm]`
— le scaling diagonal Ŝ_ffn vit sur `up_proj`. Table 9 : `W_e·W_up →
Π W* W_up Ŝ_ffn Ẑ_ffn`. RowSort élimine Ẑ (permutation) mais **pas Ŝ**
(scaling diagonal : modifie le multiset de chaque ligne). Donc `vote_up`
est ~0 % par construction, sans rapport avec la défense.

**Correctif** :
- retirer `up` de la liste par défaut des vues (`views: str = "gate"`) ;
- documenter dans le docstring pourquoi `up` est inutilisable (Ŝ non
  éliminé par RowSort) — la vue informative de la Table 9 est `gate`
  (`W_e·W_gate → Π W* W_gate Ẑ_ffn`, Ẑ tué par RowSort) ;
- le vote global entre une vue informative et une vue inerte ne prouve
  rien : voter uniquement sur les vues informatives.

## 2. Normalisation des signatures avant appariement

**Constat** : `nearest_neighbor_rows(Y, X)` en L₁ sur signatures **non
normalisées**. Le bruit α_e (embedding, `W_e_obf = (W_e + α_e·σ·noise)@P̂`)
change les normes des lignes ; un facteur d'échelle global déplace les
appariements vers les lignes de X de plus faible norme (tokens rares) →
effondrement indépendant de toute défense.

Note : le κ empirique n'apparaît PAS dans le produit direct des poids de la
VMA (il vit dans les normes de l'obfusqué, non traversées par le produit
`W̃_e·W̃_gateᵀ`) — vérifié dans chained_transform.py. Mais la normalisation
reste requise pour la robustesse au bruit α_e.

**Correctif** : normaliser chaque signature triée (L₂) avant l'appariement
dans `vma_product.py::run_vma_product` et `modal_app.py::vma_product_full` :
`Y = Y / Y.norm(dim=1, keepdim=True)` (idem X). Vérifier que le contrôle
positif reste ~100 % (la normalisation ne doit pas détruire le signal).

## 3. Agrégation : sommer les coûts au lieu du vote par mode

**Constat** : `torch.stack(preds).mode(dim=0)` = majorité stricte sur des
prédictions indépendantes ~0,5 % → le vote détruit le signal faible (le
mode de 36 quasi-aléatoires ≈ 0 %).

**Correctif** : au lieu de voter sur des prédictions, **sommer les matrices
de coût** (distances L₁ entre signatures) sur les couches, puis faire **un
unique appariement** sur la somme — plus proche d'une affectation globale,
le signal s'agrège au lieu de se noyer. API :
`per_view[view]` contient les matrices de coût cumulées (N×V) au lieu des
prédictions ; le vote devient un `argmin` global par ligne.

## 4. ISA : vocab-matching discret (avant toute conclusion)

**Constat** : loss 0,007 mais récupération 0 % = symptôme de la relaxation
continue : l'optimisation trouve des soft tokens dans l'enveloppe convexe du
simplexe qui reproduisent l'état caché sans être le prompt. Thomas et al.
concluent l'inverse (états hautement distincts) car leur recherche est
**discrète et autorégressive**.

**Correctif** : implémenter un vocab-matching discret — pour chaque position,
tester les ids permutés du vocabulaire (échantillon) contre l'état caché
cible (plus proche voisin dans l'espace des activations, ou alignement
d'embedding) — sans relaxation continue. Ne conclure « canal hidden non
informatif » qu'après ce test.

## 5. frwiki : échantillon plus grand avant de citer

**Constat** : top-1 0,809 et perplexité 1,88 anormalement bons → l'échantillon
de 4000 tokens est dominé par du balisage répétitif. Le +14 % est plausible
mais repose sur une mesure trop petite.

**Correctif** :
- augmenter `--max-tokens` (ex. 20 000) et `--n-files` ;
- filtrer le balisage (lignes courtes, titres, URLs) avant tokenisation ;
- re-mesurer la perplexité/top-1 des 3 modèles, puis mettre à jour la
  cellule 34 (journal) et STATUS.md.

## Ordre d'exécution

1. Fin du contrôle positif → interpréter (0.1 vs 0.2) ;
2. Correctifs 1-3 (vues, normalisation, agrégation) dans `vma_product.py` +
   `modal_app.py` + tests ;
3. Re-run du contrôle positif (doit rester ~100 %) + re-run VMA α_e=0,3+FT ;
4. Correctif 4 (ISA discret) — implémentation + run ;
5. Correctif 5 (frwiki) — re-mesure ;
6. Mise à jour notebook (cellules concernées) + STATUS.md + rapport.
