# Installation — obfuscator sur la machine Secretarius (démo client, sans industrialisation)

Procédure d'installation **solide mais non industrialisée** : mettre en service,
sur la machine qui héberge Secretarius (ici `sanroque`), une démo fonctionnelle
« client chiffré → LLM obfusqué servi dans le cloud » avec le schéma **h>0**
(la défense structurelle), et vérifier chaque étape.

> **Avertissements d'état réel (vérifiés le 2026-09-02) — à connaître avant
> toute démo :**
> 1. `serve()` (modal_app.py) charge `MODEL_SUBDIR = "qwen3-8b-obf"` — le
>    modèle **h=0**, que nos mesures déclarent vulnérable à ~100 % de
>    récupération VMA directe. **Ne pas démontrer ce modèle comme
>    confidentiel.** La procédure ci-dessous le remplace par le modèle h>0
>    défensif `qwen3-8b-ft-h128-a1-h02`.
> 2. Le volume Modal `obfuscator-keys` **existe encore** (recréé le
>    31/08/2026, `create_if_missing=True`) et contient `obfuscation_keys.json`,
>    contrairement à la posture affichée « volume supprimé ». L'inférence
>    (`serve`) ne le monte pas — mais un client peut vérifier. Purgez-le
>    (étape 5) et supprimez le montage du code avant toute démo.
> 3. La clé est dérivée d'une **seed** (`seed 0`, reproductibilité
>    expérimentale). Ce n'est **pas** un secret de production : pour une démo,
>    générez une clé aléatoire (`tools/rotate_pi.py`, niveau 1 de
>    `docs/rotation-cles.md`).

---

## 1. Prérequis machine

- Linux (Debian/Ubuntu testé), **Python 3.12**, `git`.
- Disque : ~20 Go libres (repo + cache HuggingFace + modèle de travail).
- RAM : 16 Go suffisent pour le **client** ; la transformation h>0 en local
  demande ~48 Go (option, étape 7) — sinon elle se fait sur Modal (GPU/CPU
  Modal, ~1-1,5 h).
- Compte [Modal](https://modal.com) authentifié (CLI) pour servir le modèle.
- Accès HuggingFace (tokenizer + modèle public Qwen3-8B).

## 2. Installation du dépôt

```bash
git clone https://github.com/mauceri/obfuscator.git
cd obfuscator
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Vérification :

```bash
.venv/bin/python -c "import torch, transformers, safetensors, modal; print(torch.__version__, transformers.__version__)"
```

Le `requirements.txt` installe `modal` dans le même venv ; si vous préférez une
CLI Modal isolée (recommandé, elle ne pollue pas le venv scientifique) :

```bash
python3.12 -m venv ~/modal-venv
~/modal-venv/bin/pip install modal
~/modal-venv/bin/modal setup        # ou : modal token new
```

> Les tests et le notebook utilisent par convention le venv scientifique
> `~/Secretarius/Wiki_LM/.venv` sur cette machine (torch/transformers déjà
> installés) — les deux sont interchangeables pour la démo.

## 3. Vérification de l'installation (aucun coût)

```bash
# tests unitaires (64 attendus)
.venv/bin/python -m pytest aloepri/tests/ -q

# notebook headless (RUN_HEAVY=False) — branches « sauté », aucun appel Modal
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/aloepri_procedure.ipynb --output /tmp/aloepri_check.ipynb
```

> **Piège** : le notebook versionne `RUN_HEAVY = True` dans sa cellule 3.
> Ne jamais lancer `nbconvert --execute` sur l'original sans avoir mis
> `RUN_HEAVY = False` (incidents de runs Modal accidentels — coût réel).

## 4. Artefacts côté client (secrets)

| Artefact | Rôle | Emplacement |
|---|---|---|
| Tokenizer Qwen3-8B | tokenize/detokenize (public) | cache HF (téléchargé au 1er usage) |
| Clé `obfuscation_keys.json` | Π, Π⁻¹ — la confidentialité | `artifacts/obfuscation_keys.json` (gitignoré) |
| Clé API du service | Bearer (fail-closed) | `~/.aloepri-api-key` + Secret Modal `aloepri-api-key` |

- La clé de permutation actuelle (`seed 0`) est dans `artifacts/obfuscation_keys.json`
  (151 936 entrées). Pour une démo, **générez une clé aléatoire** (voir
  `docs/rotation-cles.md`, niveau 1) — une démo avec `seed 0` documentée
  publiquement est une objection client immédiate.
- Créer le secret du service (une fois par compte Modal) :

```bash
~/modal-venv/bin/modal secret create aloepri-api-key ALOEPRI_API_KEY="$(openssl rand -hex 32)"
# même valeur dans ~/.aloepri-api-key (0600) pour le client
```

## 5. Purge de l'incohérence « volume de clés »

L'état actuel (volume `obfuscator-keys` recréable) contredit la posture. Pour la
démo, le plus simple sans toucher au code de transformation :

```bash
~/modal-venv/bin/modal volume delete obfuscator-keys -y   # purge l'existant
```

Puis, dans `modal_app.py`, passer `create_if_missing=False` pour `keys_vol`
(ligne ~89) afin qu'aucun run ne le recrée silencieusement. *(La clé reste de
toute façon dérivable de la seed pour la transformation — la vraie séparation
est documentée dans `docs/rotation-cles.md`.)*

## 6. Service h>0 (le modèle de la démo)

Le service `serve()` (FastAPI, `@modal.asgi_app`) charge le modèle pointé par
`MODEL_SUBDIR`. **Régler sur le modèle h>0 défensif** avant déploiement :

```bash
# modal_app.py, ligne ~40 — UNE ligne à changer pour la démo :
#   MODEL_SUBDIR = "qwen3-8b-ft-h128-a1-h02"     # h>0 défensif (α_e=1,0, α_h=0,2)
#   # (défaut : "qwen3-8b-obf" = h=0, à ne PAS démontrer comme confidentiel)
~/modal-venv/bin/modal deploy modal_app.py
```

Le modèle `qwen3-8b-ft-h128-a1-h02` doit exister sur le volume Modal
`obfuscator-models` (il y est déjà sur le compte `mauceri`). Sur un autre
compte, le téléverser :

```bash
~/modal-venv/bin/modal volume put obfuscator-models <chemin-local> qwen3-8b-ft-h128-a1-h02/
```

Vérification (health fail-closed) :

```bash
curl -s -H "Authorization: Bearer $(cat ~/.aloepri-api-key)" \
  https://<workspace>--obfuscator-aloepri-serve.modal.run/health
# → {"status":"ok"} ; sans Bearer → 401
```

## 7. Client chiffré de démonstration

`tools/client_demo.py` (ce dépôt) : tokenize public → **permutation Π locale** →
envoi des ids permutés → dépermutation → texte. Le serveur ne voit jamais un
texte ni une clé.

```bash
.venv/bin/python tools/client_demo.py \
  --url https://<workspace>--obfuscator-aloepri-serve.modal.run \
  --keys artifacts/obfuscation_keys.json \
  --api-key "$(cat ~/.aloepri-api-key)" \
  "Quelle est la capitale de la France ?"
```

Sortie attendue (verbose) : les ids **clairs** → ids **permutés** envoyés →
ids permutés reçus → dépermutés → texte. C'est le cœur démontrable :
**le transport et le serveur ne manipulent que des nombres sans signification
pour qui n'a pas Π.**

## 8. Vérification de bout en bout (gate de la démo)

1. `pytest aloepri/tests/` → 64 passed ;
2. `/health` → 200 avec Bearer, 401 sans ;
3. round-trip client : « La capitale de la France est **Paris** » correct ;
4. le verbose du client montre que la requête HTTP ne contient que des ids
   permutés (aucun fragment de texte lisible) ;
5. le modèle servi est bien h>0 (hidden 4352) : interroger `serve_env` ou
   vérifier le `config.json` du modèle sur le volume.

## 9. Pièges connus (résumés)

- `RUN_HEAVY=True` versionné → runs Modal accidentels (voir §3).
- Jupyter : après chaque réécriture du notebook, **File → Reload** (jamais
  Overwrite).
- `seed 0` = expérimental, pas un secret (voir `docs/rotation-cles.md`).
- Le service h=0 `qwen3-8b-obf` reste présent sur le volume : ne pas le servir
  pour la démo ; le retirer du volume si la démo doit être irréprochable :
  `modal volume rm -r obfuscator-models qwen3-8b-obf`.
- La transformation h>0 locale (option, sans Modal) nécessite ~48 Go RAM :
  exécuter `obfuscate_chained` depuis un venv local (pas de CLI dédiée —
  petit script d'appel, cf. cellule 27 du notebook pour les paramètres).

## Références

- Mesures de sécurité : `STATUS.md` (table des résultats + lecture sécurité).
- Procédure détaillée : notebook `notebooks/aloepri_procedure.ipynb` (journal
  daté, cellule 35).
- Rotation des clés : `docs/rotation-cles.md`.
- Présentation client : `presentations/demo-avocats.md` (Marp).
