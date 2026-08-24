# Notebook de procédure AloePri (Qwen3-8B, Modal) — design

- **Date** : 2026-08-24
- **Projet** : obfuscator — https://github.com/mauceri/obfuscator (machine `sanroque`)
- **Statut** : approuvé en session (brainstorming) — prêt pour `writing-plans`
- **Objectif (1ᵉʳ jalon d'obfuscator)** : un notebook exécutable dont les cellules
  détaillent **point par point** la procédure d'AloePri (arXiv 2603.01499) pour
  Qwen3-8B, orchestrant Modal pour les étapes lourdes.

## Décisions de design (validées en session)

| Sujet | Décision |
|---|---|
| Périmètre de la procédure | **Hybride** : procédure validée par le POC (h=0, pas de P̂/Q̂, α_e=0.3, β=8, seed 0) d'abord ; section « matrices clés P̂/Q̂ (h>0) + chaînage » réservée, à compléter quand la brique d+2h sera implémentée. |
| Exécution | **Notebook local (Jupyter) orchestrant Modal** via `app.run()` / `.remote()` — aligné sur le 2ᵉ objectif (application d'obfuscation complète + service serverless du modèle obfusqué). |
| Attaques | **Arc complet** : baseline (fuite totale) → partiel sans attention ni permutation (fuite en clair via attention) → partiel sans attention avec permutation (ids permutés récupérés) → modèle complet (défendu). Priorité de construction : variante « sans attention, avec permutation » d'abord, puis arc complet. |
| Code | **Autonome dans obfuscator** : zéro dépendance à Secretarius. Cellules pédagogiques autonomes ; étapes lourdes via un fichier compagnon `modal_app.py` (approche B). Extraction en package réservée au 2ᵉ objectif. |

## Contexte

- **POC source** (dans `~/Secretarius`, dépôt privé — **référence uniquement**, pas de
  dépendance) : `aloepri_poc/` (embedding/attention/ffn obfuscation, `key_matrix.py`,
  `block_perm.py`, `transform_streaming.py`, `client_wrapper.py`, `isa_attack.py`,
  mesures), `aloepri_modal/` (app Modal en posture stricte), `aloepri_freq_attack/`.
- **Papier** : [AloePri — Towards Privacy-Preserving LLM Inference via Covariant
  Obfuscation (arXiv 2603.01499)](https://arxiv.org/pdf/2603.01499) — Algorithme 1
  (matrices clés P̂/Q̂), §5.2.2 (embedding), §5.4 (schéma complet/chaînage), Tableau 4
  (TTRSR : 0,82 % avec matrices clés, 0 % avec permutation tête/bloc).
- **Paramètres validés par le POC** (grandeur nature) : Qwen/Qwen3-8B, seed 0,
  α_e=0.3, β=8 (Ẑ actif), bf16 (arithmétique float32), `rope_scaling=off` automatique
  (q_norm/k_norm sur Qwen3), décodage `enable_thinking=False` + greedy +
  `repetition_penalty=1.05` + blocage `<think>` (151667).
- **Artefacts absents de cette machine** : aucun modèle obfusqué ni clés en local —
  le notebook les **produit** (transform sur Modal, chemin reproductible).

## Architecture & composants

```
~/obfuscator/
├── AGENTS.md
├── README.md                    # venv, lancement du notebook, coûts Modal, sécurité
├── requirements.txt             # jupyter, numpy, torch (CPU), transformers, safetensors, modal, requests
├── .gitignore                   # clés, artefacts, volumes locaux, *.ipynb_checkpoints
├── modal_app.py                 # App Modal "obfuscator-aloepri" (graine du 2ᵉ objectif)
│   ├── transform()              # CPU : transforme Qwen3-8B (streaming) → volume obfuscator-models + clés obfuscator-keys
│   ├── serve()                  # GPU L4 : ASGI /generate (IDs permutés) + /health — posture stricte
│   ├── isa_attack()             # GPU A100-40GB : attaque ISA canal hidden/attn (ids permutés en entrée, CSV)
│   └── diag()                   # état des volumes (optionnel)
├── notebooks/
│   └── aloepri_procedure.ipynb  # LE livrable
└── docs/superpowers/specs/      # ce design + futurs plans
```

- **Kernel local** : venv dédié (torch CPU suffit ; les étapes lourdes passent par
  Modal ; torch local sert aux démos petite échelle + tokenizer client).
- **Nommage** : app et volumes préfixés `obfuscator-` (`obfuscator-models`,
  `obfuscator-keys`) — aucune collision avec le POC (`aloepri-*`).

## Structure du notebook (`notebooks/aloepri_procedure.ipynb`)

| Section | Cellules (pas à pas) | Exécution |
|---|---|---|
| **0. Setup** | Objectif, modèle de menace, références ; config (Qwen/Qwen3-8B, seed 0, α_e=0.3, β=8) ; vérification d'environnement ; flag `RUN_HEAVY` | local |
| **1. Matrices d'obfuscation** | 1.1 permutation de vocabulaire Π (déterministe par seed) ; 1.2 bruit embedding (α_e, α_h, relatif à σ(W)) ; 1.3 facteurs attention R̂ (rotation RoPE), Ĥ (diagonal — off sur Qwen3 q_norm), Ẑ (bloc, β), Û_vo (orthogonale) ; 1.4 facteurs FFN ; **vérifs : Π·Π⁻¹=Id, conditionnement des facteurs** | local, petite échelle (numpy/torch) |
| **2. Obfuscation du modèle** | 2.1 check architecture (heads, tying, q_norm/k_norm) ; 2.2 `transform.remote()` (streaming ~16 Go, shards) ; 2.3 vérification bit-à-bit (échantillons) + récupération des clés | **Modal (CPU)** |
| **3. Export Modal** | 3.1 volume `obfuscator-models` ; 3.2 déploiement `serve()` ; 3.3 health check (cold start ~1-3 min, réessais) | **Modal** |
| **4. Tests de base** | 4.1 codec client (round-trip permute/dépermute) ; 4.2 questions simples (capitale, calcul, haïku) — décodage validé | local + **Modal** |
| **5. Attaques (arc complet)** | 5.1 méthode (capture → soft tokens → gradient, 2 phases, perte relative) ; **5.2 partiel sans attn, avec permutation** (ids permutés) ; 5.3 partiel sans attn ni permutation (fuite **en clair**) ; 5.4 baseline (fuite totale) ; 5.5 modèle complet (défendu) ; 5.6 tableau comparatif + interprétation | **Modal (A100)** |
| **6. P̂/Q̂ — à compléter** | Section réservée : Algorithme 1, h>0, redimensionnement d+2h, chaînage inter-couches — cellules stub « à implémenter » (référence : spec Secretarius 2026-08-22) | — |

**Modèles de test de l'arc d'attaques** : pour les variantes (baseline, partiel, complet)
nécessitant des modèles différents, privilégier un **Qwen3 miniature** (déjà utilisé dans
les tests du POC) pour les cellules rapides, et le **vrai Qwen3-8B** pour la démonstration
finale (coût A100 maîtrisé).

## Flux de données & posture de sécurité

- Clés + tokenizer **côté client uniquement** (secret) → ids permutés → `POST /generate`
  (Modal ne reçoit que des nombres) → ids permutés → dépermutation locale → texte.
- Le serveur ne charge aucun tokenizer et ne voit aucune clé.
- Attaques : l'attaquant = opérateur serveur (poids obfusqués, aucune clé) ; les ids
  d'entrée des attaques sont les ids **permutés** calculés côté client.
- Les clés ne sont **jamais** poussées (`.gitignore`) ni laissées sur Modal : volume
  `obfuscator-keys` récupéré puis supprimé après `transform()`.

## Pièges & gestion d'erreurs (hérités du POC)

- `@modal.asgi_app()` (la fonction RETOURNE l'app FastAPI) — pas `@modal.web_server` (303).
- Secrets Modal lus via `os.environ` (pas de `.get()` sur l'objet Secret).
- `ephemeral_disk` Modal : minimum 524288 MiB (512 GiB).
- CLI `modal run` : pas d'annotation `list[int]` (ids en CSV).
- `apply_chat_template(tokenize=True)` peut renvoyer un `Encoding` — passer par
  `tokenize=False` puis `tokenizer(...)`.
- Attaque ISA : perte **relative** (MSE/variance), recuit de température + phase 2,
  GPU A100-40GB (OOM sur L4).
- Cold start : `503` pendant ~1-3 min après scale-to-zero — boucle de réessai sur `/health`.
- Versions Python locales/remote : si des fonctions Modal sont définies dans le notebook,
  versions identiques exigées (sérialisation) — contourné par `modal_app.py` (fichier).

## Vérification (TDD léger)

- Assertions dans les cellules : Π·Π⁻¹=Id, round-trip codec, (futur) P̂·Q̂=I.
- Chaque cellule lourde se termine par une vérification reproductible : health 200,
  taux de round-trip, tableau des taux d'attaque par variante.
- Flag `RUN_HEAVY` : reproductibilité sans relancer les cellules Modal (résultats
  précédents conservés/affichés).
- Le notebook entier sert de harnais de vérification du jalon (arc complet mesurable).

## Coûts (à documenter dans le README)

- Transform sur Modal : conteneur CPU, ~30-60 min (téléchargement 16 Go + transform).
- Attaques : A100-40GB (~1,5-2 $/h), quelques minutes par variante.
- Service : L4 (~0,80 $/h), facturé au temps GPU allumé (scale-to-zero).

## Hors périmètre (ce jalon)

- Implémentation des matrices clés P̂/Q̂ (h>0) + chaînage (section 6 réservée).
- Extraction du code en package `src/obfuscator/` (2ᵉ objectif).
- Attaques TFMA/SDA, Attn-IA/Gate-IA, IMA (à évaluer plus tard).
- Rotation des clés.

## Références

- Papier AloePri : arXiv 2603.01499 (Algorithme 1, §5.2.2, §5.4, Tableau 4).
- POC Secretarius : `aloepri_poc/RESULTATS_QWEN3.md`, `RESULTATS_ISA.md`,
  `aloepri_modal/README.md`, `REPRISE.md`,
  `docs/superpowers/specs/2026-08-22-aloepri-matrices-cles-design.md` (référence pour
  la section 6).
- Skills superpowers : brainstorming (ce design), puis `writing-plans` pour l'implémentation.
