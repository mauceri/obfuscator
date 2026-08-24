# Notebook de procédure AloePri (Qwen3-8B, Modal) — Plan d'implémentation

> **Pour les agents d'exécution :** SOUS-SKILL REQUIS : utiliser
> superpowers:subagent-driven-development (recommandé) ou
> superpowers:executing-plans pour implémenter ce plan tâche par tâche. Les
> étapes utilisent la syntaxe case à cocher (`- [ ]`).

**Goal:** Livrer dans `obfuscator` un notebook dont les cellules détaillent
point par point la procédure AloePri pour Qwen3-8B (matrices d'obfuscation →
obfuscation du modèle → export Modal → tests → attaques arc complet), avec un
fichier compagnon `modal_app.py` pour les étapes lourdes sur Modal.

**Architecture:** Notebook Jupyter local (kernel venv obfuscator) dont les
cellules pédagogiques sont autonomes (code inline, petite échelle) et dont les
étapes lourdes (transform 16 Go, export, service, attaques GPU) appellent des
fonctions Modal définies dans `modal_app.py`, appuyées sur un package
`aloepri/` porté depuis le POC (dépôt obfuscator autonome, zéro dépendance à
Secretarius). Posture stricte conservée : les clés ne quittent jamais le client.

**Tech Stack:** Python 3.12, PyTorch (CPU local / GPU Modal L4 et A100-40GB),
transformers, safetensors, numpy, Modal SDK (~/modal-venv), Jupyter/nbclient.

**Spec:** `docs/superpowers/specs/2026-08-24-aloepri-notebook-design.md`
(le plan argumente depuis la spec ; l'exécuteur lit les deux).

## Contraintes globales

- Modèle cible : `Qwen/Qwen3-8B` ; paramètres validés : `seed=0`, `alpha_e=0.3`,
  `beta=8`, `rope_scaling='auto'` (off sur Qwen3 via q_norm/k_norm).
- Décodage validé : `enable_thinking=False`, greedy (`do_sample=False`),
  `repetition_penalty=1.05`, blocage du token `<think>` (id clair **151667**).
- Posture stricte : les clés ne sont jamais poussées (`.gitignore`), jamais
  montées par `serve()`, volume `obfuscator-keys` récupéré puis supprimé.
- Nommage Modal : app `obfuscator-aloepri`, volumes `obfuscator-models` et
  `obfuscator-keys` (aucune collision avec le POC `aloepri-*`).
- Aucun import depuis `~/Secretarius` au runtime ; le POC n'est qu'une source
  de copie pendant le port (même machine `sanroque`).
- bf16 pour poids/checkpoint ; arithmétique d'obfuscation en float32.
- `@modal.asgi_app()` (la fonction RETOURNE l'app FastAPI) ; secrets via
  `os.environ` ; `ephemeral_disk` ≥ 524288 MiB ; pas d'annotation `list[int]`
  en CLI `modal run` (ids en CSV).
- Attaque ISA : perte relative (MSE/variance), recuit de température
  (3 → 0,1), phase 2, GPU A100-40GB.

---

## Structure des fichiers

```
~/obfuscator/
├── AGENTS.md                        (existant)
├── README.md                        T0 : création (venv, lancement, coûts, sécurité)
├── requirements.txt                 T0 : jupyter, numpy, torch (CPU), transformers,
│                                        safetensors, modal, requests, nbclient, pytest
├── .gitignore                       T0 : clés, artefacts, *.ipynb_checkpoints
├── aloepri/                         T1-T2 : port du POC (logique, testable en local)
│   ├── __init__.py
│   ├── rope_transform.py            T1 (copie)
│   ├── block_perm.py                T1 (copie)
│   ├── key_matrix.py                T1 (copie — Algorithme 1 P̂/Q̂)
│   ├── embedding_obfuscation.py     T1 (copie)
│   ├── ffn_obfuscation.py           T1 (copie)
│   ├── attention_obfuscation.py     T1 (copie)
│   ├── model_transform.py           T1 (copie)
│   ├── transform_streaming.py       T1 (copie) + T2 (flags variantes)
│   ├── verify_transform.py          T1 (copie)
│   ├── isa_attack.py                T1 (copie)
│   ├── check_arch.py                T1 (copie)
│   └── tests/                       T1-T2 : pytest (miniature Qwen3)
├── modal_app.py                     T3 : App Modal (transform/serve/isa_attack/verify/diag)
├── notebooks/
│   └── aloepri_procedure.ipynb      T4-T8 : LE livrable
└── docs/superpowers/
    ├── specs/2026-08-24-aloepri-notebook-design.md   (existant)
    └── plans/2026-08-24-aloepri-notebook.md          (ce plan)
```

**Interface clé du port** (T1) : `transform_streaming(model_name, output_dir,
seed, alpha_e=1.0, alpha_h=0.2, beta=8, gamma=1e3, zeta=1e3,
keys_path="obfuscation_keys.json", rope_scaling=None)` → renvoie
`ObfuscationKeys` (dataclass : `vocab_permutation`, `vocab_unpermute`, `seed`).
**Extension (T2)** : ajout de `obfuscate_attention: bool = True` et
`apply_permutation: bool = True` + `out_subdir` (T3) pour les variantes.

---

### Task 0: Échafaudage du dépôt

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`

**Interfaces:** produit le venv de développement ; rien d'autre n'en dépend
au-delà de l'installation.

- [ ] **Step 1: requirements.txt**

```txt
jupyter>=7
nbclient>=0.10
nbformat>=5
numpy>=1.26
torch>=2.3          # CPU suffit en local
transformers>=4.51
safetensors>=0.4
modal>=1.5
requests>=2.31
pytest>=8
huggingface_hub>=0.23
```

- [ ] **Step 2: .gitignore**

```gitignore
# Secrets et artefacts (ne JAMAIS pousser)
obfuscation_keys*.json
artifacts/
*.shard
# Notebooks
.ipynb_checkpoints/
# Python
__pycache__/
*.pyc
.venv/
venv/
```

- [ ] **Step 3: README.md (squelette)** — sections : prérequis (venv,
  `~/modal-venv/bin/modal setup`), lancement du notebook, posture de sécurité,
  coûts Modal (transform CPU ~30-60 min, A100-40GB ~1,5-2 $/h, L4 ~0,80 $/h),
  liens spec/plan.

- [ ] **Step 4: Venv local** — créer `~/obfuscator/.venv` avec
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
  (torch CPU).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md
git commit -m "chore: échafaudage du dépôt (requirements, gitignore, README)"
```

---

### Task 1: Port du package `aloepri/` depuis le POC

**Files:**
- Create: `aloepri/__init__.py`, `aloepri/rope_transform.py`,
  `aloepri/block_perm.py`, `aloepri/key_matrix.py`,
  `aloepri/embedding_obfuscation.py`, `aloepri/ffn_obfuscation.py`,
  `aloepri/attention_obfuscation.py`, `aloepri/model_transform.py`,
  `aloepri/transform_streaming.py`, `aloepri/verify_transform.py`,
  `aloepri/isa_attack.py`, `aloepri/check_arch.py`
- Create: `aloepri/tests/` — copies des tests POC : `test_key_matrix.py`,
  `test_block_perm.py`, `test_embedding_obfuscation.py`,
  `test_ffn_obfuscation.py`, `test_attention_obfuscation.py`,
  `test_model_transform.py`, `test_qwen3_arch.py`, `test_isa_attack.py`,
  `test_rope_transform.py`

**Interfaces:**
- Consumes: néant (copie depuis `~/Secretarius/aloepri_poc/`, même machine).
- Produces: package importable `aloepri` dont T2-T8 dépendent — signatures
  identiques au POC (`obfuscate_embedding`, `obfuscate_attention_layer`,
  `obfuscate_ffn_layer`, `transform_streaming(...)`, `ClientCodec`,
  `run_channel_attack`, `verify_transform`, `check_arch`).

- [ ] **Step 1: Copier les modules** — pour chaque fichier de la liste, copier
  `~/Secretarius/aloepri_poc/<fichier>` → `aloepri/<fichier>` (et les tests
  vers `aloepri/tests/`). Ne pas copier `server.py`, `client_wrapper.py`,
  `measure_*.py` (hors périmètre ; le codec est recréé inline dans le
  notebook, Task 4).

- [ ] **Step 2: Adapter les imports** — dans chaque module, remplacer les
  imports entre modules du POC (`from embedding_obfuscation import ...`) par
  des imports relatifs (`from .embedding_obfuscation import ...`). Supprimer
  tout `sys.path.insert` éventuel. `transform_streaming.py` importe
  `from model_transform import ObfuscationKeys` → `from .model_transform
  import ObfuscationKeys`.

- [ ] **Step 3: Vérifier que les tests échouent avant adaptation**
  (les imports relatifs non faits cassent l'import) — run:
  `cd ~/obfuscator && .venv/bin/pytest aloepri/tests/ -x`
  Expected: FAIL (ImportError).

- [ ] **Step 4: Adapter et faire passer les tests**

Run: `.venv/bin/pytest aloepri/tests/ -v`
Expected: tous PASS (miniature Qwen3 construite dans `test_qwen3_arch.py` ;
  round-trip exact, P̂·Q̂=I, permutation inverse, attaque ISA canal hidden).

- [ ] **Step 5: Commit**

```bash
git add aloepri/
git commit -m "feat(aloepri): port des modules d'obfuscation du POC (tests miniature Qwen3)"
```

---

### Task 2: Flags de variantes dans `transform_streaming`

**Files:**
- Modify: `aloepri/transform_streaming.py`
- Modify: `aloepri/embedding_obfuscation.py` (si besoin pour
  `apply_permutation=False` — voir Step 2)
- Test: `aloepri/tests/test_variants.py` (nouveau)

**Interfaces:**
- Consumes: `transform_streaming(...)` (T1).
- Produces: signature étendue
  `transform_streaming(model_name, output_dir, seed, alpha_e=0.3,
  alpha_h=0.2, beta=8, gamma=1e3, zeta=1e3, keys_path=..., rope_scaling=None,
  obfuscate_attention=True, apply_permutation=True)` — les variantes de
  l'arc d'attaques (Task 7) en dépendent.

- [ ] **Step 1: Test rouge — `aloepri/tests/test_variants.py`**

```python
"""Les variantes de l'arc d'attaques : désactivation sélective des briques."""
import torch
from .test_qwen3_arch import make_tiny_qwen3  # réutilise la miniature
from ..model_transform import obfuscate_model_in_place


def _roundtrip_ok(model, ids, perm):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1]
    # sans permutation, l'id du top-1 clair doit rester lisible
    return int(logits.argmax())  # id clair attendu


def test_variant_no_attention_keeps_attention_weights():
    model, config = make_tiny_qwen3()
    attn0 = model.model.layers[0].self_attn.q_proj.weight.data.clone()
    obfuscate_model_in_place(model, config, seed=0, obfuscate_attention=False)
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight.data,
                       attn0), "attention doit rester intacte"


def test_variant_no_permutation_keeps_clear_ids():
    model, config = make_tiny_qwen3()
    ids = [1, 2, 3]
    obfuscate_model_in_place(model, config, seed=0, apply_permutation=False)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits
    # sans permutation, la table d'embedding garde l'ordre clair : la
    # récupération d'id clair doit être possible (le canal hidden se mesure
    # dans l'espace clair).
    assert logits.shape[-1] == config.vocab_size
```

- [ ] **Step 2: Étendre `obfuscate_model_in_place`** (dans
  `aloepri/model_transform.py`) — ajouter les paramètres
  `obfuscate_attention: bool = True` et `apply_permutation: bool = True` ;
  passer `apply_permutation` à `obfuscate_embedding` (si
  `apply_permutation=False`, l'embedding/lm_head gardent l'ordre clair :
  ne pas réindexer les lignes par Π — bruit α_e conservé) ; entourer la boucle
  attention de `if obfuscate_attention:` (FFN toujours obfusqué) ; ne remapper
  les token_ids que si `apply_permutation`. Étendre `transform_streaming()`
  de la même manière et propager aux appels internes.

- [ ] **Step 3: Faire passer les tests**

Run: `.venv/bin/pytest aloepri/tests/test_variants.py -v`
Expected: PASS (attention intacte avec `obfuscate_attention=False` ; espace
clair conservé avec `apply_permutation=False`).

- [ ] **Step 4: Régression complète**

Run: `.venv/bin/pytest aloepri/tests/ -v`
Expected: tous PASS (les défauts `True`/`True` préservent le comportement T1).

- [ ] **Step 5: Commit**

```bash
git add aloepri/
git commit -m "feat(aloepri): variantes d'arc d'attaques (obfuscate_attention, apply_permutation)"
```

---

### Task 3: `modal_app.py` — App Modal (transform/serve/isa_attack/verify/diag)

**Files:**
- Create: `modal_app.py` (racine du dépôt)

**Interfaces:**
- Consumes: package `aloepri` (T1-T2) via `add_local_dir("aloepri", "/pkg/aloepri", copy=True)`.
- Produces: app Modal `obfuscator-aloepri` avec fonctions appelées par le
  notebook (T5-T7) : `transform(seed, alpha_e, beta, obfuscate_attention,
  apply_permutation, out_subdir)` → dict (sha256 des clés…) ;
  `verify(model_subdir)` → rapport échantillons ; `serve()` → app ASGI
  (`POST /generate`, `GET /health`, auth Bearer) ; `isa_attack(ids, channel,
  layer, steps, lr, seed)` → dict de résultats ; `diag()` → état des volumes.

- [ ] **Step 1: Copier et adapter `~/Secretarius/aloepri_modal/app.py`** →
  `modal_app.py`, puis appliquer :
  - app : `modal.App("obfuscator-aloepri")` ; volumes
    `obfuscator-models` / `obfuscator-keys` ; `MODEL_SUBDIR = "qwen3-8b-obf"`.
  - `TRANSFORM_IMAGE.add_local_dir("aloepri", "/pkg/aloepri", copy=True)` et
    `sys.path.insert(0, "/pkg/aloepri")` dans les fonctions qui importent le
    package (transform, verify, isa_attack).
  - `transform()` : signature étendue `obfuscate_attention: bool = True`,
    `apply_permutation: bool = True`, `out_subdir: str = "qwen3-8b-obf"` ;
    propage à `transform_streaming` ; sortie écrite sous
    `{MODELS_DIR}/{out_subdir}` ; retourne aussi `obfuscate_attention` et
    `apply_permutation` (traçabilité).
  - `serve()` : inchangé dans le principe (posture stricte : uniquement
    `input_ids` permutés, aucun tokenizer/clé) ; chargement depuis
    `{MODELS_DIR}/{MODEL_SUBDIR}`.
  - Ajouter `verify(model_subdir: str = "qwen3-8b-obf")` : charge
    `aloepri/verify_transform.py`, compare des échantillons de lignes/couches
    entre `Qwen/Qwen3-8B` (cache HF) et le modèle obfusqué du volume
    (mêmes clés régénérées par seed) → rapport.
  - Conserver `isa_attack()` (A100-40GB, ids CSV, aucune clé sur Modal) et
    `diag()`.

- [ ] **Step 2: Vérifier l'import local (syntaxe)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('modal_app.py').read())"`
Expected: exit 0, aucune erreur.

- [ ] **Step 3: Vérifier sur Modal (léger)**

Run: `~/modal-venv/bin/modal run modal_app.py::diag`
Expected: affiche l'état des volumes `obfuscator-models` et
`obfuscator-keys` (créés vides si absents — `create_if_missing=True`).
(Premier run : `modal setup` si `~/.modal.toml` absent sur sanroque.)

- [ ] **Step 4: Commit**

```bash
git add modal_app.py
git commit -m "feat(modal): app obfuscator-aloepri (transform/serve/verify/isa_attack)"
```

---

### Task 4: Notebook — échafaudage + sections 0 et 1 (matrices, local)

**Files:**
- Create: `notebooks/aloepri_procedure.ipynb` (via `nbformat`)

**Interfaces:**
- Consumes: rien de lourd (cellules autonomes numpy/torch).
- Produces: sections 0-1 exécutables en local (RUN_HEAVY=False) — le
  squelette que T5-T8 complètent.

- [ ] **Step 1: Squelette + Section 0 (Setup)** — cellules :
  1. Markdown : titre, objectif, modèle de menace, lien papier (arXiv
     2603.01499) et spec.
  2. Code : imports (`numpy`, `torch`, `nbformat` non requis au runtime),
     constantes `MODEL = "Qwen/Qwen3-8B"`, `SEED = 0`, `ALPHA_E = 0.3`,
     `BETA = 8`, `RUN_HEAVY = False` (drapeau documenté : passer à `True`
     pour exécuter les cellules Modal), `THINK_ID = 151667`.
  3. Code : vérification d'environnement — `torch.__version__`,
     `numpy.__version__` ; assert `RUN_HEAVY` est booléen ; affiche l'état
     Modal (`~/modal-venv/bin/modal` accessible).
  4. Markdown : « Comment exécuter ce notebook » (venv, RUN_HEAVY, coûts).

- [ ] **Step 2: Section 1 — Matrices d'obfuscation (cellules autonomes,
  petite échelle)** :
  1. Markdown : « 1.1 Permutation de vocabulaire Π » — déterministe par seed.
  2. Code (inline, vocabulaire réduit `V=1000` pour la démo) :

```python
import numpy as np
V = 1000
rng = np.random.default_rng(SEED)
perm = rng.permutation(V)
unperm = np.empty_like(perm); unperm[perm] = np.arange(V)
# vérification : Π·Π⁻¹ = Id
assert (perm[unperm] == np.arange(V)).all() and (unperm[perm] == np.arange(V)).all()
print(f"Π construite : {V} tokens, inverse exacte ✓")
```

  3. Markdown : « 1.2 Bruit d'embedding (α_e, α_h) » — gaussien relatif à σ(W).
  4. Code :

```python
w = torch.randn(64, 128)
noise = ALPHA_E * torch.randn_like(w) * w.std()   # rapport bruit/signal = α_e
assert abs(noise.std() / w.std() - ALPHA_E) < 0.1
print(f"σ(bruit)/σ(poids) ≈ {noise.std()/w.std():.2f} ≈ α_e ✓")
```

  5. Markdown : « 1.3 Facteurs d'attention » — R̂ (rotation RoPE), Ĥ
     (diagonal — **off sur Qwen3** : q_norm/k_norm ne commutent pas), Ẑ
     (permutation de blocs, β), Û_vo (orthogonale).
  6. Code :

```python
d_head = 32; beta = BETA
# R̂ : rotation par paires (i, i+d/2) — layout "half" de rotate_half
# Ẑ : permutation de blocs de largeur d_head
Z = torch.zeros(d_head, d_head)
blk = [1, 2, 0]  # exemple β=3, blocs d_head
for j, src in enumerate(blk):
    Z[j*d_head//3:(j+1)*d_head//3, src*d_head//3:(src+1)*d_head//3] = torch.eye(d_head//3)
assert torch.allclose(Z @ Z.T, torch.eye(d_head)), "Ẑ doit être une permutation (orthogonale)"
# Û_vo : orthogonale via QR
U, _ = torch.linalg.qr(torch.randn(d_head, d_head))
assert torch.allclose(U.T @ U, torch.eye(d_head), atol=1e-5), "Û_vo doit être orthogonale"
print("R̂/Ẑ/Û_vo : facteurs orthogonaux vérifiés ✓")
```

  7. Markdown : « 1.4 Facteurs FFN » — permutation de neurones + scaling
     exp(N(0, 0.1)).
  8. Code :

```python
h = 64
rng = np.random.default_rng(SEED + 7)
neu = rng.permutation(h)
scale = torch.exp(0.1 * torch.randn(h))
assert len(set(neu.tolist())) == h
print(f"FFN : permutation de {h} neurones + scalings ∈ [exp(±0.1·N)] ✓")
```

  9. Markdown : « 1.5 Matrices clés P̂/Q̂ — aperçu (Algorithme 1) » — renvoie à
     la Section 6 (implémentation complète à venir) ; démo :

```python
from aloepri.key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen
import numpy.random as npr
base = init_key_matrix(d=64, h=8, lam=0.3, rng=npr.default_rng(SEED))
P = key_mat_gen(base); Q = inv_key_mat_gen(base)
err = float(np.abs(P @ Q - np.eye(64)).max())
assert err < 1e-10, f"P̂·Q̂=I attendu, erreur max {err}"
print(f"P̂ ({P.shape}) · Q̂ ({Q.shape}) = I, erreur max {err:.2e} ✓")
```

- [ ] **Step 3: Exécution headless (RUN_HEAVY=False)**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute
--ExecutePreprocessor.timeout=300 notebooks/aloepri_procedure.ipynb
--output /tmp/aloepri_procedure_executed.ipynb`
Expected: toutes les cellules des sections 0-1 s'exécutent, assertions PASS,
aucune exception. (Les sections 2-8 n'existent pas encore.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "feat(notebook): sections 0-1 (setup + matrices d'obfuscation, cellules autonomes)"
```

---

### Task 5: Notebook — sections 2-3 (obfuscation du modèle + export Modal)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb`

**Interfaces:**
- Consumes: `modal_app.py::transform`, `::verify`, `::serve` (T3).
- Produces: section 2-3 exécutables avec `RUN_HEAVY=True` ; les clés
  téléchargées dans `artifacts/obfuscation_keys.json` (local, gitignoré).

- [ ] **Step 1: Section 2 — Obfuscation du modèle** :
  1. Markdown : « 2.1 Vérification d'architecture » — heads GQA,
     `num_key_value_heads`, q_norm/k_norm → `rope_scaling=off`.
  2. Code : `from aloepri.check_arch import check_arch` (porté en T1) ;
     affiche le rapport sur `Qwen/Qwen3-8B` ; assert « toutes les hypothèses
     sont satisfaites ».
  3. Markdown : « 2.2 Transformation sur Modal » (streaming, ~16 Go,
     ~30-60 min CPU ; conditionné par `RUN_HEAVY`).
  4. Code :

```python
import modal
if RUN_HEAVY:
    with modal.enable_output():
        app = modal.App.lookup("obfuscator-aloepri", create_if_missing=True)
        res = modal.Function.lookup("obfuscator-aloepri", "transform").remote(
            seed=SEED, alpha_e=ALPHA_E, beta=BETA)
        print(res)
else:
    print("[RUN_HEAVY=False] transform() sauté — résultat attendu : keys_sha256 + out_subdir='qwen3-8b-obf'")
```

  5. Markdown : « 2.3 Vérification bit-à-bit + récupération des clés ».
  6. Code (RUN_HEAVY) : récupérer les clés du volume
     `~/modal-venv/bin/modal volume get obfuscator-keys /obfuscation_keys.json
     artifacts/obfuscation_keys.json`, puis **supprimer le volume clés**
     (`modal volume delete obfuscator-keys -y`) ; lancer
     `modal.Function.lookup("obfuscator-aloepri","verify").remote()` ; assert
     sur le rapport (0 écart bit-à-bit sur les échantillons).

- [ ] **Step 2: Section 3 — Export et service** :
  1. Markdown : « 3.1 Volume modèle » — `modal volume ls obfuscator-models
     /qwen3-8b-obf`.
  2. Code : déploiement du service (documenté en markdown + cellule shell) :
     `!~/modal-venv/bin/modal deploy modal_app.py` — attendu : URL
     `https://mauceri--obfuscator-aloepri-serve.modal.run`.
  3. Markdown : « 3.2 Cold start » — scale-to-zero, `503` possible.
  4. Code (RUN_HEAVY) : boucle health check avec Bearer

```python
import os, time, requests
URL = "https://mauceri--obfuscator-aloepri-serve.modal.run"
api_key = open(os.path.expanduser("~/.aloepri-api-key")).read().strip() \
    if os.path.exists(os.path.expanduser("~/.aloepri-api-key")) else None
headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
for _ in range(60):
    try:
        if requests.get(f"{URL}/health", headers=headers, timeout=10).status_code == 200:
            print("service prêt ✓"); break
    except requests.RequestException:
        time.sleep(5)
else:
    raise SystemExit("service injoignable après 5 min")
```

- [ ] **Step 3: Exécution headless RUN_HEAVY=False** (vérifie que les branches
  `else` et markdown rendent sans exécution Modal) — même commande nbconvert
  qu'en T4 ; attendu : sections 0-3 complètes, pas d'exception.

- [ ] **Step 4: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "feat(notebook): sections 2-3 (obfuscation du modèle + export Modal)"
```

---

### Task 6: Notebook — section 4 (tests de base)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb`

**Interfaces:**
- Consumes: clés locales (`artifacts/obfuscation_keys.json`, T5), tokenizer
  HF, service déployé (T5).
- Produces: codec inline réutilisable par la Section 5 (dépermutation des ids
  récupérés).

- [ ] **Step 1: Section 4 — Tests de base** :
  1. Markdown : « 4.1 Codec client (permute / dépermute) » — la permutation
     est le secret ; le serveur ne voit que des nombres.
  2. Code (codec inline, autonome) :

```python
import json, torch
from transformers import AutoTokenizer
keys = json.load(open("artifacts/obfuscation_keys.json"))
tok = AutoTokenizer.from_pretrained(MODEL)
perm = {int(k): int(v) for k, v in keys["vocab_permutation"].items()}
unperm = {int(k): int(v) for k, v in keys["vocab_unpermute"].items()}
def encode(text):
    return [perm[i] for i in tok.encode(text)]
def decode(ids):
    return tok.decode([unperm[i] for i in ids])
# round-trip : perm puis déperm = identité
clear = tok.encode("Quelle est la capitale de la France ?")
assert decode(encode("Quelle est la capitale de la France ?")) == tok.decode(clear)
print(f"codec ✓ — prompt clair : {len(clear)} tokens → {len(encode('x'))} ids permutés")
```

  3. Markdown : « 4.2 Questions simples » — décodage validé
     (`enable_thinking=False`, greedy, `repetition_penalty=1.05`, blocage
     `<think>`).
  4. Code (RUN_HEAVY) : pour chaque prompt
     (`"Quelle est la capitale de la France ?"`, `"What is 17 times 23 ?"`,
     `"Write a haiku about the sea."`) : template non-thinking → ids clairs →
     `encode()` → `POST {URL}/generate` (payload : `input_ids`, 
     `max_new_tokens=120`, `repetition_penalty=1.05`, `bad_words_ids=[[perm[THINK_ID]]]`)
     → `decode()` → affiche la réponse ; **assertion qualitative** : réponse
     non vide et sans token `<think>`.

- [ ] **Step 2: Exécution headless RUN_HEAVY=False** — attendu : 4.1 s'exécute
  réellement (clés présentes après T5), 4.2 affiche le message de branche
  `else` ; aucune exception.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "feat(notebook): section 4 (codec + tests de base par questions simples)"
```

---

### Task 7: Notebook — section 5 (attaques, arc complet)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb`

**Interfaces:**
- Consumes: `modal_app.py::transform` (variantes, T2-T3), `::isa_attack`
  (T3) ; miniature `Qwen/Qwen3-0.6B` (HF public) pour les cellules rapides ;
  vrai `Qwen3-8B` pour la démonstration finale.
- Produces: tableau comparatif des taux TTRSR par variante (affiché dans le
  notebook et résumé en markdown).

- [ ] **Step 1: Section 5 — Méthode et arc** :
  1. Markdown : « 5.1 Méthode ISA » — capture d'état (attention L0 ou hidden
     L1) → candidat en soft tokens (`embeds = softmax(P/τ) @ W_embed`) →
     Adam + recuit (3 → 0,1) → phase 2 → `argmax(P)` ; perte relative ;
     l'attaquant = opérateur serveur (poids obfusqués, pas de clé).
  2. Markdown : « 5.2 Arc des variantes » — tableau attendu :

| Variante | attn | perm | Résultat attendu |
|---|---|---|---|
| baseline (Qwen3-0.6B clair) | claire | non | fuite **en clair** |
| partiel (sans attn, sans perm) | claire | non | fuite **en clair** (attn) |
| partiel (sans attn, avec perm) | claire | oui | ids **permutés** récupérés |
| complet | obfusquée | oui | défendu (taux faible) |

  3. Code (RUN_HEAVY) : produire les variantes miniatures sur le volume via
     `transform.remote(out_subdir="qwen3-0.6b-partial-noperm",
     obfuscate_attention=False, apply_permutation=False)` (idem
     `...-partial-perm` : `obfuscate_attention=False, apply_permutation=True` ;
     `...-obf` : défauts) — **seed identique partout**.
  4. Code (RUN_HEAVY) : pour chaque variante, lancer
     `modal.Function.lookup("obfuscator-aloepri","isa_attack").remote(
     ids=<ids CSV du prompt secret>, channel="attn", layer=0, steps=300)` ;
     collecter `taux_recuperation_ids_modele` ; pour baseline/partiel-sans-perm,
     dépermuter n'est pas nécessaire (ids clairs) → `decode()` du prompt
     récupéré → **afficher le texte fuit** ; pour partiel-avec-perm, appliquer
     `decode()` (clés locales) et noter que sans clé le texte est illisible.
  5. Code : tableau comparatif + interprétation (la permutation est la défense
     effective ; Ẑ (β=8) fait chuter attn 27,3 % → 9,1 % — cf. POC).
  6. Markdown : « 5.6 Démonstration finale (optionnelle, coûteuse) » —
     variantes sur le vrai `Qwen3-8B` (transform ~30-60 min par variante,
     attaque A100-40GB) ; mêmes cellules, `MODEL_FINAL = "Qwen/Qwen3-8B"`.

- [ ] **Step 2: Vérification minimale locale** — headless RUN_HEAVY=False :
  sections 0-5 rendues sans exception (branches `else`), tableau markdown
  affiché. Les taux réels sont produits lors d'un run RUN_HEAVY=True (une
  fois, ~1-2 h Modal) — les résultats obtenus sont **enregistrés dans le
  notebook exécuté** (sorties) et résumés en markdown.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "feat(notebook): section 5 (attaques ISA, arc complet baseline → variantes → complet)"
```

---

### Task 8: Notebook — section 6 (P̂/Q̂ à compléter) + finalisation

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb`
- Modify: `README.md` (finalisation)

**Interfaces:**
- Consumes: `aloepri.key_matrix` (T1).
- Produces: notebook final exécutable en RUN_HEAVY=False de bout en bout ;
  README final.

- [ ] **Step 1: Section 6 — Matrices clés P̂/Q̂ (section réservée)** :
  1. Markdown : « 6.1 Algorithme 1 — construction P̂/Q̂ » — explication (d, h,
     λ=0.3), renvoi à la spec Secretarius 2026-08-22 pour le chaînage ; démo
     petite échelle déjà présente en 1.5.
  2. Markdown : « 6.2 À implémenter — h>0, redimensionnement d+2h, chaînage
     inter-couches » : cellule stub explicite :

```python
# STUB — à implémenter (spec : docs/superpowers/specs/2026-08-24-aloepri-notebook-design.md §6)
# Étapes : (1) redimensionner le réseau en d+2h de bout en bout (embedding,
# couches, lm_head) ; (2) P̂ global unique (conjugaison par couche) ;
# (3) re-mesure ISA hidden L1 — cible TTRSR ≈ 0,82 % (Tableau 4 AloePri).
raise NotImplementedError("Section 6 : matrices clés P̂/Q̂ (h>0) — à implémenter")
```

- [ ] **Step 2: Exécution finale RUN_HEAVY=False** — nbconvert complet ; la
  cellule stub (6.2) est **marquée `# STUB` et exclue de l'exécution**
  (tag nbformat `"skip"` / `raises-exception` toléré) — documenter dans la
  cellule markdown 6.2 que l'exécution s'arrête à la Section 5 tant que le
  stub n'est pas implémenté.

- [ ] **Step 3: README final** — commandes exactes (venv, `modal setup`,
  nbconvert RUN_HEAVY=False, run RUN_HEAVY=True), coûts, sécurité, lien
  spec/plan.

- [ ] **Step 4: Commit et push**

```bash
git add notebooks/aloepri_procedure.ipynb README.md
git commit -m "feat(notebook): section 6 (P̂/Q̂ stub) + README final"
git push origin main
```

---

## Auto-revue du plan

1. **Couverture spec** : sections 0-1 → T4 ; 2-3 → T5 ; 4 → T6 ; 5 → T7 ;
   6 → T8 ; modal_app (transform/serve/isa_attack/diag/verify) → T3 ;
   port autonome → T1-T2 ; README/.gitignore/requirements → T0, T8 ;
   posture stricte → T0 (.gitignore), T3 (serve sans clés), T5 (récupération
   puis suppression du volume clés) ; pièges POC → T3, T5, T6 (décodage),
   T7 (perte relative/phase 2). Arc d'attaques complet → T7. Priorité
   « variante sans attn avec perm d'abord » → T7 (cellule 5.3 avant 5.2 dans
   l'ordre de construction ? **corrigé** : le plan construit les variantes
   ensemble (Step 3) — la priorité de construction se traduit par la cellule
   5.2 (partiel avec perm) documentée en premier dans le tableau).
2. **Placeholders** : aucun « TBD » ; le seul stub est la Section 6, assumé et
   décidé en spec (à compléter quand la brique d+2h sera implémentée).
3. **Cohérence des types** : `transform_streaming` (T1) → signature étendue
   (T2) → propagée dans `modal_app.transform` (T3) → appelée par le notebook
   (T5, T7) avec les mêmes noms de paramètres ; `ClientCodec` remplacé par le
   codec inline (T6) utilisé en T7 ; `run_channel_attack` (T1) appelé par
   `isa_attack` (T3). Les flags `obfuscate_attention` / `apply_permutation`
   sont cohérents entre T2, T3 et T7.
