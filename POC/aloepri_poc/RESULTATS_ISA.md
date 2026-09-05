# Résultats — Attaques ISA par gradient sur Qwen3-8B obfusqué (Modal)

Méthode et résultats mesurés en grandeur nature (2026-08-21) sur le modèle
obfusqué servi sur Modal (`aloepri_modal/app.py::isa_attack`, GPU A100-40GB).

Sources : AloePri arXiv 2603.01499 (Appendix D.1, Tableau 4) et
« Depth Gives a False Sense of Privacy: LLM Internal States Inversion »,
arXiv 2507.16372 (méthode d'inversion par optimisation en deux phases).

## Modèle de menace

L'attaquant est **l'opérateur du serveur** : il possède les poids obfusqués
et observe tout ce qui se passe pendant l'inférence d'une requête (états
cachés, scores d'attention). Il **n'a pas** la clé de permutation (côté
client).

## Méthode (implémentée dans `aloepri_poc/isa_attack.py`)

1. **Capture** : pour le prompt secret (IDs permutés, l'entrée réelle du
   modèle), on enregistre un état interne — état caché d'une couche
   (`channel=hidden`) ou pondérations d'attention (`channel=attn`).
2. **Paramétrisation** : le candidat X2 est représenté par des **logits par
   position** P ∈ R^{T×vocab} ; l'entrée du modèle est
   `embeds = softmax(P/τ) @ W_embed` (différentiable → le gradient remonte
   jusqu'à P à travers la table d'embedding).
3. **Phase 1** : Adam + recuit de température (τ : 3 → 0,1) sur une perte
   **relative** MSE/variance (les états cachés d'un vrai LLM ont des
   amplitudes énormes ; une MSE brute écrase le gradient).
4. **Phase 2** (2507.16372) : ré-initialisation des logits près de l'argmax
   de la phase 1 + optimisation à température basse — corrige les choix
   discrets figés prématurément par le recuit.
5. **Résultat** : ids_récupérés = argmax(P) ; métrique = taux de
   correspondance avec les IDs réellement envoyés au modèle.

## Résultats — β=1 (modèle servi initial, Ẑ_block = identité)

Prompt secret de **22 tokens** (templaté non-thinking, IDs permutés) ;
l'attaque n'utilise que les poids obfusqués — les clés ne servent qu'au
harnais de mesure.

| Canal | Couche | Taux de récupération des ids du modèle | Loss (rel.) |
|---|---|---|---|
| hidden | 1 (peu profonde) | **95,5 %** (21/22) | 0,020 |
| hidden | 18 (profonde) | 4,5 % (1/22) | 0,093 |
| attn | 0 | 27,3 % (6/22) | 1,05 (divergente) |

## Résultats — β=8 (défense restaurée, 2026-08-22)

| Canal | Couche | Taux de récupération des ids du modèle | Loss (rel.) |
|---|---|---|---|
| hidden | 1 (peu profonde) | **90,9 %** (20/22) | 0,044 |
| attn | 0 | **9,1 %** (2/22) | 1,17 (divergente) |

**Effet de la restauration** : la permutation de blocs Ẑ (β=8) fait chuter
la récupération via attention d'un facteur ~3 (27,3 % → 9,1 %) et dégrade
légèrement le canal hidden (95,5 % → 90,9 %, les états cachés héritant de la
perturbation). Le canal hidden récupère toujours ~91 % des ids du modèle —
mais ce sont les ids PERMUTÉS : sans la clé, aucun texte.

(Modèle jouet Qwen3 miniature, même attaque : canal hidden 100 % baseline et
obfusqué ; canal attn sous-déterminé — les taux d'attention ne suffisent pas
à inverser les ids à cette échelle.)

## Interprétation

1. **L'attaque par gradient fonctionne** : les états cachés PEU PROFONDS
   s'inversent presque parfaitement (95,5 %), conformément au papier
   2507.16372 (« low-depth ISs »). Les états profonds résistent (4,5 %) —
   le canal attention est faible avec cette méthode (27 %), comme sur le
   modèle jouet (sous-détermination).

2. **Ce que l'attaquant récupère, ce sont les IDs PERMUTÉS** — l'entrée que
   le modèle a réellement vue. Sans la clé de permutation (jamais sur le
   serveur), il ne peut pas les traduire en texte : **la permutation est la
   défense effective contre ISA** dans notre schéma (la récupération à
   95,5 % porte sur des ids permutés, pas sur le texte clair).

3. **Le bruit d'embedding (α_e) ne suffit pas à protéger les ids** à faible
   profondeur (95,5 % de récupération malgré α_e=0.3) — cohérent avec le
   Tableau 4 d'AloePri (le bruit seul laisse 40 % de TTRSR ; ce sont les
   protections d'attention qui font tomber le taux à 0 %).

4. **Défense restaurée (2026-08-22)** : le modèle servi a été re-transformé
   avec **β=8** (Ẑ_block actif) — la récupération via attention passe de
   27,3 % à 9,1 %, au prix d'une qualité quasi inchangée (le POC Qwen2.5
   mesurait +19,1 % de perplexité à β=8 vs +18,7 % à β=1 : ~0,4 pt).

## Ce qui n'est pas encore mesuré

- Comparaison **baseline vs obfusqué** sur le vrai modèle (l'attaque sur la
  baseline récupérerait des ids CLAIRS — le texte fuirait ; sur l'obfusqué
  il récupère des ids permutés). À faire pour quantifier précisément le
  « coût » de la défense.
- Attaque attention multi-couches / autre loss (la méthode actuelle diverge).
- TFMA/SDA et Attn-IA/Gate-IA (hors périmètre de cette session).

## Fichiers

- `aloepri_poc/isa_attack.py` — moteur d'attaque (module + CLI).
- `aloepri_poc/tests/test_isa_attack.py` — canal hidden robuste sur modèle
  jouet (récupère les ids du modèle ; ids permutés sur l'obfusqué).
- `aloepri_modal/app.py::isa_attack` — démonstration grandeur nature sur le
  modèle réel (ids permutés passés en argument, aucune clé sur Modal).
