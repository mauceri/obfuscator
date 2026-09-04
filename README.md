# obfuscator

Notebook de procédure AloePri (arXiv 2603.01499) pour Qwen3-8B, orchestrant
Modal pour les étapes lourdes. Dépôt autonome : zéro dépendance à Secretarius
(POC utilisé uniquement comme référence).

- [Installation / démo client](docs/installation-demo.md)
- [Runbook de reprise après sinistre](docs/DISASTER-RECOVERY.md)
- [Rotation des clés (2 niveaux)](docs/rotation-cles.md)
- [Présentation client (Marp, public avocats)](presentations/demo-avocats.md)

## Prérequis

- Python 3.12.
- Venv local :

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

- CLI Modal (`~/modal-venv/bin/modal setup`) — uniquement pour les étapes
  lourdes sur Modal (transform, export, service, attaques ISA).

## Vérification headless (RUN_HEAVY=False)

Sans authentification Modal, le notebook s'exécute de bout en bout : sections
0-1 complètes, sections 2-5 en branches « sauté » (aucun appel Modal), la
cellule stub 6.2 lève volontairement `NotImplementedError` — tolérée par le
tag nbformat `raises-exception`. Commande (depuis la racine du dépôt) :

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 notebooks/aloepri_procedure.ipynb \
  --output /tmp/aloepri_check.ipynb
```

Attendu : exit 0, aucune exception (hors la cellule stub marquée).

## Exécution complète (RUN_HEAVY=True)

1. Passer `RUN_HEAVY = True` dans la cellule setup du notebook.
2. Authentifier la CLI : `~/modal-venv/bin/modal token new` (ou
   `~/modal-venv/bin/modal setup`).
3. Lancer Jupyter **depuis la racine du dépôt** :

   ```bash
   .venv/bin/jupyter notebook notebooks/aloepri_procedure.ipynb
   ```

   puis exécuter les cellules dans l'ordre (ou relancer la commande nbconvert
   ci-dessus avec `RUN_HEAVY=True`).

Le run complet : transform du modèle (~16 Go, ~30-60 min CPU Modal),
vérification bit-à-bit, récupération puis **suppression** du volume clés,
déploiement du service, tests de base (section 4, décodage validé), attaques
ISA de l'arc complet (section 5, ~1-2 h A100-40GB). Les clés sont
téléchargées dans `artifacts/obfuscation_keys.json` (gitignoré) puis le
volume clés Modal est supprimé. La section 6 (matrices clés P̂/Q̂, h>0) est
réservée : cellule stub « à implémenter ».

## Matrices clés h>0 (schéma complet, commit `aef5859`)

La section 6 est maintenant implémentée (`aloepri/chained_transform.py`,
tests `tests/test_chained_transform.py`) : chaînage global P̂/Q̂ (§5.4) avec
reconstruction du modèle à `hidden_size = d + 2h`, Wnorm fusionnée dans les
couches adjacentes, normes à poids scalaire κ (empirique par couche — la
variante gaussienne du papier biaise l'échelle). Le round-trip devient
**approximatif** (erreur κ, §5.2.5) : la vérification est un contrôle de
qualité (`verify_chained` : corrélation logits + top-1), plus du bit-à-bit.

Deux corrections à l'Algorithme 1 du papier découvertes à l'implémentation :
(a) F1/F2 ~ N(0, 1/d) (pas 1/h) ; (b) lignes de C scalées par √(h/d) — sans
quoi ‖xP̂‖/‖x‖ ≈ 1.45 et le round-trip diverge par couches.

**Sécurité** : la VMA directe (embedding vs table claire) devient
structurellement impossible (d+2h ≠ d) — cf. `artifacts/vma_report.md` ; le
bruit α_e ne défend plus que la VMA produit (Table 9). **Qualité** : à
α_e=0.3 le bruit embed casse la génération du 0.6B (marges serrées) ; le 8B
(d=4096, plus robuste) est le juge — α à ajuster selon `verify_chained`.
Déploiement 8B : `modal run modal_app.py::transform_chained --seed 0
--alpha-e 0.3 --alpha-h 0.2 --h 128 --out-subdir qwen3-8b-obf-h128` puis
`verify_chained`.

## Posture de sécurité

Les clés d'obfuscation (`obfuscation_keys*.json`) ne quittent **jamais** le
client : elles ne sont ni poussées (voir `.gitignore`) ni transmises à Modal
(`serve()` ne monte aucun volume clés et ne charge aucun tokenizer ;
l'attaquant ISA n'a que les poids obfusqués, sans la clé de permutation).

Le secret API `aloepri-api-key` est **requis** (fail-closed) : le créer avant
de déployer le service avec
`~/modal-venv/bin/modal secret create aloepri-api-key ALOEPRI_API_KEY=<valeur>`
(la même valeur que le fichier `~/.aloepri-api-key` côté client). Sans ce
secret, `serve()` refuse **toutes** les requêtes : `/generate` et `/health`
renvoient 401.

## Coûts Modal

- Transform CPU : ~30-60 min (Qwen3-8B) ; variantes Qwen3-0.6B : quelques
  minutes.
- Attaques ISA : A100-40GB, ~1,5-2 $/h, quelques minutes par variante.
- Service : L4, ~0,80 $/h, facturé au temps GPU allumé (scale-to-zero).


## Correctif Qwen3 (2026-08-24)

Les RMSNorm de tête `q_norm`/`k_norm` de Qwen3 portent un γ appris **non
constant** qui ne commute pas avec les rotations/permutations denses de
`head_dim` (R̂/Ẑ) : `γ ⊙ (R̂·x) ≠ R̂·(γ ⊙ x)`. Le round-trip logits des vrais
Qwen3 était cassé (corr 0,35) et la génération dégénérait en charabia — les
modèles jouets (γ=1) et Qwen2.5 (pas de q_norm) masquaient le défaut. Depuis
le commit `9f6355e`, `rope_rotation=False` est automatique sous `q_norm` :
R̂/Ẑ = identité ; restent exacts les permutations de têtes, Û_vo (v/o), la
permutation de vocabulaire et le FFN. La défense d'attention se réduit alors
au mélange de têtes + Û_vo — la **permutation de vocabulaire reste la
protection effective du texte** (contre ISA ; **mais pas contre VMA** — un
attaquant ayant le modèle public récupère Π à ~100 % depuis la table
d'embedding, cf. `artifacts/vma_report.md` : la vraie défense du papier est
les matrices clés h>0, non implémentées ici).

## Liens spec / plan

- [Spec](docs/superpowers/specs/2026-08-24-aloepri-notebook-design.md)
- [Plan](docs/superpowers/plans/2026-08-24-aloepri-notebook.md)
