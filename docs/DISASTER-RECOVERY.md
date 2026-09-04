# Runbook — Reconstruction des modèles obfusqués sur Modal (reprise après sinistre)

Procédure **pas à pas, vérifiable**, pour reconstruire l'état de production
(Qwen3-14B et Qwen3-8B obfusqués servis sur Modal) après une perte des
volumes Modal, des modèles obfusqués ou de la machine locale — **GitHub
intact** (source de vérité du code et de la procédure).

> Exécutable par un agent avec accès shell et compte Modal. Chaque étape a
> un **critère de succès** : ne pas passer à la suivante s'il n'est pas
> atteint. Budget total : ~5-7 h (dont ~4-5 h d'attente Modal),
> ~15-25 $.

## 0. Ce qui survit / ce qui est perdu

| Ressource | Survit ? | Source |
|---|---|---|
| Code (`modal_app.py`, `tools/`, notebooks, docs) | ✅ | GitHub `mauceri/obfuscator` |
| Paramètres et mesures | ✅ | notebook `aloepri_procedure.ipynb` (journal), `STATUS.md`, `docs/` |
| Modèles **clairs** Qwen3-14B/8B | ✅ | HuggingFace (public) |
| Clé Π (permutation) | ✅ *dérivable* | `random.Random(seed=0).shuffle` — seed documentée |
| Modèles **obfusqués** | ❌ | volume Modal (à re-créer par ce runbook) |
| Fine-tuné GEPA (8B) | ❌ | volume Modal (supprimé — à re-créer si l'identique 8B est requis) |
| Proxies systemd + config Tailscale | ⚠️ | machine locale `sanroque` (à re-créer si perdue, §8) |

## 1. Prérequis

```bash
# accès GitHub (repo cloné) + HF (token dans ~/.cache/huggingface/token)
git clone https://github.com/mauceri/obfuscator.git && cd obfuscator
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# CLI Modal authentifiée + secret du service (fail-closed)
python3.12 -m venv ~/modal-venv && ~/modal-venv/bin/pip install modal
~/modal-venv/bin/modal setup
~/modal-venv/bin/modal secret create aloepri-api-key ALOEPRI_API_KEY="$(openssl rand -hex 32)"
echo "ALOEPRI_API_KEY=<même valeur>" > ~/.config/obfuscator/proxy.env && chmod 600 ~/.config/obfuscator/proxy.env

# venv scientifique (tests/proxy) — sinon ~/Secretarius/Wiki_LM/.venv
```

**Critère** : `.venv/bin/python -c "import modal, transformers, torch"` OK ;
`modal secret list` montre `aloepri-api-key`.

## 2. Régénérer la clé Π (seed 0 — expérimentale)

La clé est **déterministe** : `random.Random(0).shuffle(range(V))`, V=151 936.

```bash
.venv/bin/python - <<'EOF'
import json, random
V = 151936
rng = random.Random(0)
permuted = list(range(V)); rng.shuffle(permuted)
perm = dict(zip(range(V), permuted))
unperm = {v: k for k, v in perm.items()}
assert all(unperm[perm[t]] == t for t in range(1000))
json.dump({"vocab_permutation": {str(k): v for k, v in perm.items()},
           "vocab_unpermute": {str(k): v for k, v in unperm.items()},
           "seed": 0}, open("artifacts/obfuscation_keys.json", "w"))
print("clé régénérée (seed 0,", V, "entrées)")
EOF
```

**Critère** : le fichier `artifacts/obfuscation_keys.json` existe (151 936
entrées) ; `git status` ne le montre pas (gitignoré).

> ⚠️ La seed 0 est expérimentale/documentée. Pour un usage sensible,
> générer une clé aléatoire APRÈS reconstruction (voir
> `docs/rotation-cles.md` + `notebooks/rotation_pi.ipynb`).

## 3. (Optionnel) Fine-tune GEPA 8B — requis pour l'IDENTIQUE du 8B

Le 8B obfusqué `qwen3-8b-ft-h128-a1-h02` actuel est construit depuis le
fine-tuné GEPA. Pour le reproduire **à l'identique**, refaire le FT d'abord :

```bash
~/modal-venv/bin/modal run modal_app.py::finetune_corpus \
    --model-name Qwen/Qwen3-8B --epochs 5 --batch-size 8 \
    --seq-len 128 --lr 2e-5 --out-subdir qwen3-8b-ft-gepa
```
A100-80GB, ~83 min, ~4-8 $. Sortie : volume `obfuscator-models/qwen3-8b-ft-gepa`.

**Critère** : `modal volume ls obfuscator-models qwen3-8b-ft-gepa` non vide.

> **Alternative simplifiée** : le fine-tune n'apportait rien (mesures 8B :
> la VMA ne dépend pas du FT, et il dégradait la qualité). On peut obfusquer
> la base 8B **directement** (étape 5B) et mettre à jour les références
> (`serve_8b`, proxy, notebook) vers `qwen3-8b-base-h128-a1-h02`.

## 4. Obfusquer le Qwen3-14B (h>0, α_e=1,0/α_h=0,2)

```bash
~/modal-venv/bin/modal run modal_app.py::transform_chained \
    --seed 0 --alpha-e 1 --alpha-h 0.2 --h 128 \
    --model-name Qwen/Qwen3-14B --out-subdir qwen3-14b-h128-a1-h02
```
CPU Modal 128 Go (déjà calibré dans le code), ~1-1,5 h, ~4-8 $.

**Critère** : `modal volume ls obfuscator-models qwen3-14b-h128-a1-h02`
contient `model.safetensors` ; le `config.json` a `hidden_size: 5376`.

## 5. Obfusquer le Qwen3-8B

**5A — depuis le FT (identique)** :
```bash
~/modal-venv/bin/modal run modal_app.py::transform_chained \
    --seed 0 --alpha-e 1 --alpha-h 0.2 --h 128 \
    --model-name /models/qwen3-8b-ft-gepa --out-subdir qwen3-8b-ft-h128-a1-h02
```
**5B — base directe (simplifié, recommandé si pas de FT)** :
```bash
~/modal-venv/bin/modal run modal_app.py::transform_chained \
    --seed 0 --alpha-e 1 --alpha-h 0.2 --h 128 \
    --model-name Qwen/Qwen3-8B --out-subdir qwen3-8b-base-h128-a1-h02
# puis mettre à jour : serve_8b → qwen3-8b-base-h128-a1-h02 (modal_app.py),
# proxy (outil systemd) et notebook (MODEL 8b)
```
~1-1,5 h, ~2-4 $.

**Critère** : `config.json` du modèle obfusqué a `hidden_size: 4352` (8B).

## 6. Vérifications de qualité (avant déploiement)

```bash
# round-trip approximatif (corr logits > 0,95 attendu)
~/modal-venv/bin/modal run modal_app.py::verify_chained \
    --model-subdir qwen3-14b-h128-a1-h02 --base-ref Qwen/Qwen3-14B
# tests unitaires locaux
.venv/bin/python -m pytest aloepri/tests/ -q        # 64 passed
```
**Critère** : corrélation logits ≥ 0,95 ; capitale → « Paris » ; 64 tests verts.

## 7. Déployer les services (14B sur A100-40GB, 8B sur L4)

```bash
~/modal-venv/bin/modal deploy modal_app.py
```
Crée deux endpoints web : `...-serve.modal.run` (14B) et
`...-serve-8b.modal.run` (8B).

**Critère** :
```bash
curl -H "Authorization: Bearer $(cat ~/.aloepri-api-key)" \
  https://mauceri--obfuscator-aloepri-serve.modal.run/health      # {"status":"ok"}
curl -H "Authorization: Bearer $(cat ~/.aloepri-api-key)" \
  https://mauceri--obfuscator-aloepri-serve-8b.modal.run/health   # {"status":"ok"}
```

## 8. Proxies locaux + Tailscale (si la machine est perdue)

Venv scientifique requis (`fastapi`, `uvicorn`, `requests`, `transformers`) —
sinon : `pip install fastapi "uvicorn[standard]" requests transformers`.

Unité systemd utilisateur (×2, port 8001 → 14B, 8002 → 8B) :
`~/.config/systemd/user/obfuscator-proxy.service` et
`obfuscator-proxy-8b.service` — cf. `docs/acces-openai.md` pour le contenu
exact (ExecStart = `tools/openai_proxy.py --keys artifacts/... --url
https://...-serve...modal.run --port 8001/8002`).

```bash
systemctl --user daemon-reload
systemctl --user enable --now obfuscator-proxy obfuscator-proxy-8b
# exposition tailnet (ports HTTPS dédiés, 443 déjà pris)
tailscale serve --bg --https=9443 http://127.0.0.1:8001   # 14B
tailscale serve --bg --https=9444 http://127.0.0.1:8002   # 8B
```
**Critère** : `curl http://127.0.0.1:8001/health` et
`https://sanroque.tailc69141.ts.net:9443/health` répondent.

## 9. Vérification de bout en bout

```bash
# chat chiffré direct (ids permutés → réponse déchiffrée)
.venv/bin/python tools/client_demo.py \
    --url https://mauceri--obfuscator-aloepri-serve.modal.run \
    --keys artifacts/obfuscation_keys.json \
    --api-key "$(cat ~/.aloepri-api-key)" "Quelle est la capitale de la France ?"
# via le proxy (protocole OpenAI) — attendu « Paris. »
curl http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b-h128-a1-h02","messages":[{"role":"user","content":"Quelle est la capitale de la France ?"}],"max_tokens":40}'
```
**Critère final** : réponse correcte **et** la preuve d'obfuscation tient
(ids clairs envoyés directement → non-sens, voir le test A/B du notebook).

## 10. Pièges connus (à connaître avant de lancer)

1. **`transform_chained`** : ~128 Go RAM CPU (calibré dans le code) ; ne pas
   réduire.
2. **Le service ne s'arrête pas sans `stop_token_id`** : l'eos du config est
   l'id clair (jamais émis par le modèle obfusqué). Le proxy fournit l'id
   permuté de `<|im_end|>` (151645) — ne pas l'omettre dans un client maison.
3. **Mode thinking Qwen3** : utiliser le chat template
   `enable_thinking=False` (le proxy le fait) — sinon le modèle « pense » à
   voix haute en continuation libre.
4. **Volume de clés supprimé** : plus aucune écriture de clés sur Modal
   (posture) ; la clé est dérivée de la seed côté client.
5. **Ne pas re-supprimer les modèles gardés** : après reconstruction, ne
   garder sur le volume que les modèles servis (voir nettoyage).
6. **Jupyter** : File → Reload après toute réécriture de notebook ; jamais
   `nbconvert --execute` avec `RUN_HEAVY=True` (runs Modal accidentels).

## 11. Budget récapitulatif (ordre de grandeur)

| Étape | Durée | Coût |
|---|---|---|
| FT GEPA 8B (option identique) | ~1,5 h A100-80GB | ~4-8 $ |
| Transform 14B | ~1-1,5 h CPU | ~4-8 $ |
| Transform 8B | ~1-1,5 h CPU | ~2-4 $ |
| Vérifications + déploiement + proxies | ~1 h | ~1-3 $ |
| **Total** | **~5-7 h** (dont ~4-5 h d'attente) | **~10-20 $** |
