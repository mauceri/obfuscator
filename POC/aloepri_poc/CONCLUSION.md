# Conclusion — POC AloePri sur Qwen3-8B (déploiement Modal)

- **Date** : 2026-08-22
- **Périmètre** : POC complet AloePri (arXiv 2603.01499) porté sur
  Qwen3-8B, servi en serverless sur Modal (posture stricte : zéro clé côté
  serveur), attaques ISA par gradient construites et mesurées.
- **Références détaillées** : `RESULTATS_QWEN3.md`, `RESULTATS_ISA.md`,
  `REPRISE.md`, `docs/superpowers/specs/2026-08-22-aloepri-matrices-cles-design.md`.

## 1. Bilan du POC

**Ce qui est validé de bout en bout :**

- **Mécanique** : permutation de vocabulaire + bruit d'embedding + attention/
  FFN obfusqués — round-trip exact (logits préservés modulo la permutation),
  transform bit-identique local ↔ Modal, vérification par échantillons.
- **Qualité** (grandeur nature, α_e=0.3, β=8) : réponses correctes et
  cohérentes (capitale, calcul 391, haïku, recette, multi-tours). Coût
  mesuré sur le POC Qwen2.5 : +13,6 % à +19,1 % de perplexité selon α_e.
- **Déploiement** : serveur Modal qui ne reçoit que des IDs permutés, aucun
  tokenizer ni clé ; client local (codec, proxy OpenAI-compatible, serveur
  notebook). Volume des clés supprimé de Modal.
- **Sécurité mesurée (ISA)** : la permutation protège le texte (l'attaquant
  ne récupère que des ids permutés) ; Ẑ (β=8) fait chuter la récupération
  via attention (27,3 % → 9,1 %) ; le canal hidden (90,9 % à L1) est la
  faiblesse restante, ciblée par l'étape matrices clés (P̂/Q̂, cible 0,82 %
  du papier).

## 2. La piste AloePri vaut-elle la peine d'être poussée plus avant ?

**Oui, il est fondé de la pousser — en tant que piste de recherche, avec un
regard lucide sur ses limites.**

Arguments pour :

1. Le mécanisme est **validé de bout en bout** (exactitude du round-trip,
   qualité acceptable, déploiement serverless réel, posture stricte
   fonctionnelle).
2. La défense est **mesurable et directionnellement conforme au papier** :
   chaque brique (permutation, Ẑ, matrices clés à venir) a un effet
   quantifiable sur une attaque réelle.
3. Le modèle de menace est **réel et motivant** pour Secretarius :
   confidentialité d'une question isolée face à un opérateur curieux ou
   compromis — le canal pertinent (ISA) est précisément celui que le POC a
   instrumenté.
4. Les prochaines étapes sont **bien définies** : matrices clés (0,82 %),
   comparaison baseline, rotation/économie des clés, analyse LoRA (ci-après).

Limites à garder à l'esprit (honnêteté scientifique) :

1. **Pas de garantie formelle** : la défense est heuristique ; la propriété
   Rényi-DP du papier (§6) n'a pas été vérifiée, et h=0 a perdu la propriété
   d'expansion.
2. **Coût qualité réel** (+13-19 % de perplexité) ; l'arbitrage sécurité/
   qualité (α_e) est central.
3. **Rotation des clés non résolue** : re-transformer le modèle pour changer
   la permutation est coûteux, et la rotation ne protège pas rétroactivement
   une requête déjà capturée contre ISA.
4. **Attaques non épuisées** : TFMA/SDA (trafic), Attn-IA/Gate-IA (poids),
   IMA (inversion entraînée) restent à évaluer.
5. Les **chiffres du papier (0 %, 0,82 %, 87 %) n'ont pas été reproduits**
   exactement — nos mesures sont directionnelles.

Verdict : pousser la piste (matrices clés puis évaluation formelle) avant
toute affirmation de garantie de confidentialité en production.

## 3. Piste LoRA — analyse à mener (et réponse technique)

**Question posée** : est-il possible d'obfusquer des adaptateurs LoRA et de
les adjoindre « à la demande » au modèle de base obfusqué, avec exactement
les mêmes paramètres ?

**Réponse technique : OUI, en principe — par linéarité.** Toutes les
transformations d'obfuscation du POC sont **linéaires** dans les poids
(vérifié dans le code : permutations de lignes/colonnes, scalings
diagonaux, facteurs orthogonaux R̂/Ĥ/Ẑ/Û_vo, permutations de têtes,
permutation de vocabulaire). Or un adaptateur LoRA est une somme
W′ = W + (α/r)·B·A. La transformation T commute avec l'addition :
**T(W′) = T(W) + T(ΔW)** — il suffit donc de transformer l'adaptateur
ΔW = B·A par le même T pour que le modèle obfusqué + adaptateur transformé
se comporte EXACTEMENT comme le modèle original + adaptateur original
(modulo la permutation de vocabulaire).

De plus, T **factorise à travers la décomposition bas-rang** : selon la
projection cible, seul B ou A est transformé (ex. gate_proj : lignes
permutées/scalées → T(B)·A ; down_proj : colonnes → B·T(A) ; attention q :
facteur gauche a_qᵀ → (a_qᵀ·B)·A ; o : facteur droit → B·(A·Û_vo⁻ᵀ)).
**Le rang et la structure de l'adaptateur sont préservés** — mêmes
paramètres (r, α, modules cibles), seuls les poids changent.

Conditions pour que ce soit vrai en pratique :

1. **Transformation côté client** : le détenteur des clés transforme
   l'adaptateur hors-ligne (les facteurs par couche sont régénérables depuis
   la seed — à formaliser dans `ObfuscationKeys`) ; le serveur ne reçoit que
   l'adaptateur obfusqué (posture stricte identique au modèle de base).
2. **Adjonction à la demande** : LoRA se fusionne à l'inférence — plusieurs
   adaptateurs pré-transformés peuvent être chargés/échangés sans toucher au
   modèle de base (le serveur doit exposer un choix d'adaptateur).
3. **Adaptateurs sur lm_head/embedding** : leurs lignes doivent subir la
   permutation de vocabulaire (Π) comme le modèle de base.
4. **Alternative plus simple** : entraîner le LoRA **directement sur le
   modèle obfusqué** — l'adaptateur naît dans l'espace obfusqué, aucune
   transformation nécessaire (mais l'entraîneur doit alors accéder au modèle
   obfusqué ; la confidentialité du jeu de données d'entraînement devient une
   question de politique, pas de mécanique).
5. **Exactitude héritée** : les approximations du modèle de base (Ẑ avec
   β>1, commutation q_norm avec rope_scaling off, arrondi bf16) s'appliquent
   à l'identique à l'adaptateur — l'adaptateur n'ajoute pas d'inexactitude
   nouvelle au-delà de celle du base.

**Ce qu'il faudra valider** (analyse dédiée, dans la lignée des tests du
POC) : sur modèle jouet — base obfusquée + adaptateur transformé == base
originale + adaptateur original (logits exacts modulo permutation), pour
chaque projection cible (q/k/v/o, gate/up/down, éventuellement
embedding/lm_head) ; puis en grandeur nature (qualité d'un adaptateur LoRA
réel avant/après transformation).
