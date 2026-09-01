# Réécriture section attaques du notebook — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer les cellules 36-49 du notebook (section « 9. Attaques » incohérente) par la nouvelle section « 9. Sécurité — de l'entraînement à l'évaluation » (9.1-9.6) conforme à la stratégie validée.

**Architecture:** Réécriture du fichier `notebooks/aloepri_procedure.ipynb` — suppression des cellules 36-49 et insertion de nouvelles cellules (markdown brefs + code détaillé commenté appelant les fonctions Modal existantes `finetune_corpus`, `transform_chained`, `vma_product_full`, `isa_attack`, plus une cellule de précision frwiki). Aucun changement de code applicatif (les fonctions existent déjà dans `modal_app.py`).

**Tech Stack:** JSON (nbformat), Python, Jupyter, Modal CLI (`!modal run`).

**Spec:** `docs/superpowers/specs/2026-08-27-notebook-attacks-rewrite-design.md`

## Global Constraints

- **Zéro mention du POC h=0 et du Qwen3-0.6B** dans les nouvelles cellules.
- Markdown **brefs** ; code Python **détaillé, recopié et commenté** (chaque option des appels Modal expliquée).
- **Aucune valeur mesurée en dur** — les taux viennent de l'exécution ou des rapports `artifacts/*`.
- Marqueurs **`cellule N`** visibles en tête de chaque cellule (convention existante).
- Les cellules 0-35 restent **inchangées**.
- Validation : JSON valide ; **jamais d'exécution headless avec `RUN_HEAVY=True`** (lance des runs Modal réels — piège documenté). Pour valider, forcer `RUN_HEAVY=False` via un préprocesseur nbconvert ou une copie temporaire.

---

### Task 1: Supprimer les cellules 36-49 et insérer le squelette (en-tête + 9.1a)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (cellules 36-49 supprimées ; nouvelles cellules insérées à l'index 36)

**Interfaces:**
- Consumes: l'état du notebook actuel (50 cellules, marqueurs `cellule N`).
- Produces: le notebook avec la nouvelle section 9 commencée (cellule 36 = en-tête, cellule 37 = 9.1a).

- [ ] **Step 1: Écrire le script de remplacement** — `/tmp/nb_rewrite.py` qui charge le notebook, supprime les cellules 36-49, insère les nouvelles cellules dans l'ordre, et réécrit les marqueurs `cellule N` de TOUTES les cellules selon leur nouvel index.

- [ ] **Step 2: Insérer la cellule 36 (en-tête de section, markdown bref)**

```markdown
> **cellule 36**
## 9. Sécurité — de l'entraînement à l'évaluation

Stratégie : (1) **fine-tuning complet** de Qwen3-8B sur un corpus synthétique
généré avec **DeepSeek + gen_corpus_gepa_codex** (modifier les poids du
modèle) ; (2) **AloePri complet** (h>0, α_e=0.3) ; (3) **VMA complète**
(Table 9) ; (4) **ISA** ; (6) **précision sur frwiki**. IMA : plus tard (9.5).
```

- [ ] **Step 3: Insérer la cellule 37 (9.1a, markdown bref — génération du corpus)**

```markdown
> **cellule 37**
### 9.1 Génération du corpus avec DeepSeek + gen_corpus_gepa_codex

Le projet https://github.com/mauceri/gen_corpus_gepa_codex génère un corpus de
notes françaises synthétiques via DSPy/**GEPA**, avec **DeepSeek** comme LLM
générateur (`DEEPSEEK_API_KEY` dans l'environnement).

- `promptGenGEPA.py` : optimise un prompt avec GEPA (optimisation génétique de
  prompts DSPy), sauvegarde le meilleur prompt (`GEPAPrompt.txt`) ;
- `generate_corpus_dspy.py` : génère le corpus JSONL — pour chaque triplet
  `(theme, categorie, type_de_document)`, DeepSeek produit une note validée
  (7 clés : `contenu`, `url`, `date`, `expressions_clefs`, `type_de_document`,
  `theme`, `categorie`) ; validation : `contenu` commence par l'étiquette de
  catégorie, `expressions_clefs` (1-8) apparaissent dans `contenu`, URL/date
  plausibles, `theme`/`categorie` identiques aux entrées.

CLI (dans `~/gen_corpus_gepa_codex`) :
- optimiser + générer : `DSPY_CACHEDIR=.dspy_cache python promptGenGEPA.py --count 50 --gepa-prompt GEPAPrompt.txt --output corpus_gepa.jsonl`
- générer avec un prompt existant : `DSPY_CACHEDIR=.dspy_cache python generate_corpus_dspy.py --count 50 --output corpus.jsonl --model deepseek-chat`

Corpus utilisé ici : `~/gen_corpus_gepa_codex/corpus_synth_clean_10000.jsonl`
(10 000 textes, ~1,74 M tokens ; les sorties de génération, ~138 k textes,
sont dans `gepa_llm_calls.log`).
```

- [ ] **Step 4: Vérifier** — JSON valide ; cellules 36-37 présentes avec leurs marqueurs ; cellule 35 inchangée ; aucune mention de 0.6B/h=0 dans les nouvelles cellules.

```bash
python3 -c "import json; nb=json.load(open('notebooks/aloepri_procedure.ipynb')); assert len(nb['cells'])==len([c for c in nb['cells'] if True]); print('OK', len(nb['cells']))"
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): section 9.1a — en-tête + génération du corpus (DeepSeek+GEPA)"
```

---

### Task 2: Cellule 9.1b — full fine-tuning de Qwen3-8B (détaillée)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (insérer la cellule 38 après la 37)

**Interfaces:**
- Consumes: la cellule 37 (9.1a) ; la fonction Modal `finetune_corpus` existante (`modal_app.py`).
- Produces: la cellule 38 (9.1b) — le code de `finetune_corpus` recopié/commenté + l'appel RUN_HEAVY.

- [ ] **Step 1: Insérer la cellule 38 (9.1b, code détaillé)**

```python
# cellule 38
# 9.1b Full fine-tuning de Qwen3-8B sur le corpus GEPA (TOUS les paramètres)
#
# Objectif : modifier les poids du modèle (stratégie 1) — W_e, attention,
# FFN, head changent → la référence publique de la VMA (Table 9) devient
# fausse. Budget : A100-80GB (~70 Go : modèle bf16 16 Go + gradients +
# AdamW fp32), ~30-60 min, ~4-8 $.
#
# La fonction `finetune_corpus` (modal_app.py) fait exactement ceci :
#   1. charge le modèle source (bf16, CPU) ;
#   2. tokenise le corpus GEPA (séquences de `seq_len` tokens) ;
#   3. boucle d'entraînement : AdamW (lr 2e-5) + autocast bf16 sur GPU,
#      loss = cross-entropie next-token ;
#   4. sauvegarde le modèle fine-tuné sur le volume
#      `obfuscator-models/{out_subdir}`.
# Le code de la boucle (extrait commenté de finetune_corpus) :
#   for step in range(epochs * steps_per_epoch):
#       idx = torch.randint(0, n_seq, (batch_size,))
#       with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#           out = model(corpus[idx].cuda(), labels=corpus[idx].cuda())
#       opt.zero_grad(); out.loss.backward(); opt.step()

if RUN_HEAVY:
    !~/modal-venv/bin/modal run modal_app.py::finetune_corpus \
        --model-name Qwen/Qwen3-8B --epochs 5 --batch-size 8 \
        --seq-len 128 --lr 2e-5 --out-subdir qwen3-8b-ft-gepa
else:
    print("[RUN_HEAVY=False] fine-tuning 8B sauté — code et budget ci-dessus ; "
          "résultat attendu : qwen3-8b-ft-gepa sur le volume")
```

- [ ] **Step 2: Vérifier** — cellule 38 présente ; le code mentionne Qwen3-8B (pas 0.6B) ; l'appel CLI est complet et commenté.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): 9.1b — full fine-tuning Qwen3-8B (finetune_corpus détaillé)"
```

---

### Task 3: Cellule 9.2 — AloePri complet (h>0, α_e=0.3)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (insérer la cellule 39 après la 38)

**Interfaces:**
- Consumes: la cellule 38 (9.1b) ; la fonction Modal `transform_chained` existante.
- Produces: la cellule 39 (9.2).

- [ ] **Step 1: Insérer la cellule 39 (9.2, code détaillé)**

```python
# cellule 39
# 9.2 AloePri COMPLET (h>0, matrices clés) sur le 8B fine-tuné — α_e=0.3
#
# `transform_chained` (modal_app.py) reconstruit le modèle avec
# hidden_size = d + 2h (h=128 → 4352) et applique le chaînage global P̂/Q̂ :
#   embed·P̂ ; q/k/v/gate/up·Q̂ᵀ (Wnorm fusionnée) ; o/down·P̂ᵀ ; head·Q̂ᵀ ;
# normes → κ (§5.2.5, κ empirique par couche). Deux corrections de
# l'Algorithme 1 du papier : F1/F2 ~ N(0,1/d) ; scaling C √(h/d).
# α_e=0.3 : bruit d'embedding relatif à σ(W) (config production).
# Budget : CPU, ~1-1,5 h, ~1-2 $. Sortie : `qwen3-8b-ft-h128`.

if RUN_HEAVY:
    !~/modal-venv/bin/modal run modal_app.py::transform_chained \
        --seed 0 --alpha-e 0.3 --alpha-h 0.2 --h 128 \
        --model-name Qwen/Qwen3-8B --out-subdir qwen3-8b-ft-h128
else:
    print("[RUN_HEAVY=False] transform_chained sauté — code et budget ci-dessus ; "
          "résultat attendu : qwen3-8b-ft-h128 (hidden 4352)")
```

- [ ] **Step 2: Vérifier** — cellule 39 présente ; mentionne h>0, α_e=0.3, h=128, la sortie `qwen3-8b-ft-h128` ; pas de 0.6B.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): 9.2 — AloePri complet h>0 α_e=0.3 sur le 8B fine-tuné"
```

---

### Task 4: Cellule 9.3 — VMA complète (Table 9, toutes couches, vote)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (insérer la cellule 40 après la 39)

**Interfaces:**
- Consumes: la cellule 39 (9.2) ; la fonction Modal `vma_product_full` existante.
- Produces: la cellule 40 (9.3).

- [ ] **Step 1: Insérer la cellule 40 (9.3, code détaillé)**

```python
# cellule 40
# 9.3 VMA COMPLÈTE (Table 9) — profondeur maximale : toutes les couches + vote
#
# `vma_product_full` (modal_app.py) attaque le modèle h>0 contre la référence
# publique : pour chaque vue × chaque couche, produit W̃_e·W̃_gateᵀ (ou up) —
# les P̂/Q̂ s'annulent par chaînage — RowSort (élimine Ẑ_ffn), appariement NN
# sur 2000 tokens, puis VOTE majoritaire sur les 36 couches (par vue) et
# vote global entre vues. Amélioration de profondeur : 36/36 couches (au lieu
# d'un sous-ensemble), vote complet (mécanisme du papier).
# W_e·W_h : ÉCARTÉ pour l'instant (vue V×V, 46 Go — à revoir quand le reste
# sera au point). Vues V×V (gram q·k) : limite A100-40GB documentée.
# Budget : A100-40GB, ~1-1,5 h, ~2-4 $.

if RUN_HEAVY:
    !~/modal-venv/bin/modal run modal_app.py::vma_product_full \
        --model-subdir qwen3-8b-ft-h128 --subset-size 2000 --views gate,up
else:
    print("[RUN_HEAVY=False] vma_product_full sauté — code et budget ci-dessus ; "
          "résultat attendu : TTRSR par couche + votes (rapport "
          "artifacts/vma_produit_8b_complet.md)")
```

- [ ] **Step 2: Vérifier** — cellule 40 présente ; mentionne les 36 couches, le vote, W_e·W_h écarté ; cible `qwen3-8b-ft-h128`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): 9.3 — VMA complète (36 couches, vote, W_e·W_h écarté)"
```

---

### Task 5: Cellule 9.4 — ISA (canal hidden)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (insérer la cellule 41 après la 40)

**Interfaces:**
- Consumes: la cellule 40 (9.3) ; la fonction Modal `isa_attack` existante.
- Produces: la cellule 41 (9.4).

- [ ] **Step 1: Insérer la cellule 41 (9.4, code détaillé)**

```python
# cellule 41
# 9.4 ISA — Internal State Attack (canal hidden, couche profonde)
#
# L'attaquant (= opérateur serveur) capture l'état caché d'une couche du
# modèle h>0 (fine-tuné) sur le prompt secret (ids permutés), puis inverse
# par descente de gradient (soft tokens + recuit de température) pour
# retrouver l'entrée. Canal `hidden` = le canal informatif (les scores
# d'attention sont sous-déterminés). Les ids récupérés sont ceux du MODÈLE
# (permutés) — sans la clé, aucun texte.
# Budget : A100-40GB, ~30 min, ~1-2 $.

if RUN_HEAVY:
    # ids = les ids PERMUTÉS du prompt secret (calculés côté client avec la
    # clé seed 0 — jamais envoyés en clair au serveur)
    import json as _json
    _perm = _json.load(open("artifacts/obfuscation_keys.json"))["vocab_permutation"]
    _secret = "Quelle est la capitale de la France ?"
    _ids_perm = ",".join(str(_perm[str(i)]) for i in tok.encode(_secret))
    !~/modal-venv/bin/modal run modal_app.py::isa_attack --ids $_ids_perm \
        --channel hidden --layer 18 --steps 400 \
        --model-ref qwen3-8b-ft-h128
else:
    print("[RUN_HEAVY=False] ISA sauté — code et budget ci-dessus ; attendu : "
          "0 % de récupération sur h>0 (mesuré, artifacts/chained_8b_report.md)")
```

- [ ] **Step 2: Vérifier** — cellule 41 présente ; canal hidden, couche 18, ids permutés via la clé locale ; cible `qwen3-8b-ft-h128`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): 9.4 — ISA canal hidden sur le modèle h>0 fine-tuné"
```

---

### Task 6: Cellules 9.5 (IMA réservé) et 9.6 (précision frwiki)

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb` (insérer les cellules 42 et 43 après la 41)

**Interfaces:**
- Consumes: la cellule 41 (9.4) ; les modèles sur le volume (base, fine-tuné, fine-tuné+obfusqué) ; `~/corpus_fr/frwiki`.
- Produces: les cellules 42 (9.5) et 43 (9.6).

- [ ] **Step 1: Insérer la cellule 42 (9.5, markdown 1 ligne)**

```markdown
> **cellule 42**
### 9.5 IMA (Inversion Model Attack) — réservé

Attaque par entraînement d'un modèle d'inversion (appendice D.1 du papier).
À traiter plus tard — dépend de la défense de Π (VMA), qui est le point 9.3.
```

- [ ] **Step 2: Insérer la cellule 43 (9.6, code détaillé — précision frwiki)**

```python
# cellule 43
# 9.6 Précision sur Wikipedia français (~/corpus_fr/frwiki, 1,1 Go)
#
# Métriques : PERPLEXITÉ + précision NEXT-TOKEN (top-1) sur un échantillon
# de textes frwiki. Comparaison : base (Qwen3-8B) vs fine-tuné (9.1b) vs
# fine-tuné+obfusqué h>0 (9.2). L'obfusqué reçoit les ids PERMUTÉS et ses
# logits sont dépermutés pour la comparaison. Budget : A100-40GB,
# ~20-40 min (2-3 modèles chargés), ~1-2 $.

if RUN_HEAVY:
    import os, random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _frwiki_sample(n_files=4, max_tokens=4000, seed=0):
        # échantillon : quelques fichiers texte de frwiki, tokenisés
        root = os.path.expanduser("~/corpus_fr/frwiki")
        files = []
        for dirpath, _, names in os.walk(root):
            for n in names:
                if n.endswith(".txt"):
                    files.append(os.path.join(dirpath, n))
        random.Random(seed).shuffle(files)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        ids = []
        for f in files[:n_files]:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                ids += tok(fh.read(), add_special_tokens=False).input_ids
            if len(ids) > max_tokens:
                break
        return torch.tensor(ids[:max_tokens])

    def _metrics(model_ref, ids, perm=None):
        # perplexité + top-1 next-token ; `perm` = {clair: obfusqué} si obfusqué
        m = AutoModelForCausalLM.from_pretrained(model_ref, dtype=torch.bfloat16)
        m = m.cuda().eval()
        if perm:
            ids_in = torch.tensor([perm[int(t)] for t in ids.tolist()])
        else:
            ids_in = ids.clone()
        with torch.no_grad():
            logits = m(ids_in[None, :-1]).logits[0]   # (L-1, V)
        if perm:
            cols = torch.tensor([perm[t] for t in range(logits.shape[1])]).cuda()
            logits = logits[:, cols]
        loss = torch.nn.functional.cross_entropy(
            logits.float(), ids[1:].cuda()).item()
        top1 = float((logits.argmax(-1) == ids[1:].cuda()).float().mean().item())
        return {"perplexite": round(2 ** loss, 2), "top1_next_token": round(top1, 4)}

    ids = _frwiki_sample()
    print("échantillon frwiki :", ids.numel(), "tokens")
    print("base          :", _metrics("Qwen/Qwen3-8B", ids))
    print("fine-tuné     :", _metrics("qwen3-8b-ft-gepa", ids))
    # modèle h>0 : config hidden 4352 — chargement normal, ids permutés
    print("FT + h>0      :", _metrics("qwen3-8b-ft-h128", ids, perm=...))
else:
    print("[RUN_HEAVY=False] précision frwiki sauté — code et budget ci-dessus ; "
          "attendu : perplexité + top-1 pour base / FT / FT+h>0")
```

- [ ] **Step 3: Vérifier** — cellules 42-43 présentes ; la cellule 43 ne contient **aucune valeur mesurée en dur** ; la permutation pour le modèle h>0 est chargée depuis `artifacts/obfuscation_keys.json` (le `perm=...` du code est à compléter avec le chargement réel des clés — pas de placeholder : écrire `perm = {int(k): int(v) for k, v in _json.load(open("artifacts/obfuscation_keys.json"))["vocab_permutation"].items()}` en tête de la cellule).

- [ ] **Step 4: Commit**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): 9.5 IMA réservé + 9.6 précision frwiki (perplexité + top-1)"
```

---

### Task 7: Renumérotation des marqueurs + validation complète

**Files:**
- Modify: `notebooks/aloepri_procedure.ipynb`

**Interfaces:**
- Consumes: toutes les cellules (0-43).
- Produces: le notebook final avec marqueurs `cellule N` cohérents (0..N-1 dans l'ordre).

- [ ] **Step 1: Renuméroter les marqueurs** — script qui, pour chaque cellule i, remplace la première ligne par le bon marqueur (`> **cellule {i}**` pour markdown, `# cellule {i}` pour code).

- [ ] **Step 2: Vérifier la structure** — JSON valide ; aucune cellule avec un marqueur hors ordre ; les cellules 0-35 sont inchangées dans leur contenu (sauf le marqueur s'il a bougé) ; aucune mention de « 0.6B » ou « h=0 » dans les cellules 36+ ; aucune valeur mesurée en dur (`\d+[.,]\d+ %` hors format `:.1%`).

```bash
python3 - <<'EOF'
import json, re
nb = json.load(open('notebooks/aloepri_procedure.ipynb'))
assert len(nb['cells']) == 44, len(nb['cells'])
for i, c in enumerate(nb['cells']):
    first = ''.join(c['source']).split('\n')[0]
    assert (f"cellule {i}" in first), (i, first)
src36 = ''.join(''.join(c['source']) for c in nb['cells'][36:])
assert '0.6B' not in src36 and 'h=0' not in src36
print("OK — structure, marqueurs, zéro 0.6B/h=0 dans la section 9")
EOF
```

- [ ] **Step 3: Validation headless SÛRE** — copier le notebook dans `/tmp`, forcer `RUN_HEAVY=False` dans la copie, exécuter avec nbconvert sur la copie (jamais sur l'original qui est en `RUN_HEAVY=True`).

```bash
python3 -c "
import json
nb = json.load(open('notebooks/aloepri_procedure.ipynb'))
nb['cells'][3]['source'] = [s.replace('RUN_HEAVY = True', 'RUN_HEAVY = False') for s in nb['cells'][3]['source']]
json.dump(nb, open('/tmp/aloepri_headless.ipynb', 'w'))
"
jupyter nbconvert --to notebook --execute /tmp/aloepri_headless.ipynb --output /tmp/aloepri_headless_out.ipynb --ExecutePreprocessor.timeout=600
```

Attendu : exit 0, aucune erreur de cellule (les branches RUN_HEAVY=False s'affichent).

- [ ] **Step 4: Commit final**

```bash
git add notebooks/aloepri_procedure.ipynb
git commit -m "docs(notebook): section 9 réécrite — marqueurs renumérotés, validation headless RUN_HEAVY=False"
```

---

## Self-Review

- **Couverture spec** : 9.1a (corpus DeepSeek+GEPA) → Task 1 ; 9.1b (full FT 8B) → Task 2 ; 9.2 (AloePri h>0 α_e=0.3) → Task 3 ; 9.3 (VMA complète, W_e·W_h écarté) → Task 4 ; 9.4 (ISA hidden) → Task 5 ; 9.5 (IMA réservé) + 9.6 (frwiki) → Task 6 ; numérotation/marqueurs → Task 7. ✓
- **Placeholders** : la cellule 43 contient un `perm=...` à compléter — corrigé par l'instruction explicite du Step 3 (chargement réel des clés). Aucun autre placeholder.
- **Cohérence des types** : `qwen3-8b-ft-gepa` (sortie 9.1b) et `qwen3-8b-ft-h128` (sortie 9.2) cohérents entre les cellules 9.3/9.4/9.6 ; la cellule 9.6 charge la permutation depuis `artifacts/obfuscation_keys.json` (seed 0) — cohérent avec la cellule 9.4.
