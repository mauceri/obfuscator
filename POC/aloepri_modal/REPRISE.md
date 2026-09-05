# REPRISE — comment reprendre le travail AloePri après interruption

Point d'entrée unique pour une nouvelle session (agent frais). Lire ce fichier
AVANT tout, puis `aloepri_poc/RESULTATS_QWEN3.md`, `aloepri_poc/RESULTATS_ISA.md`
et `aloepri_modal/README.md`. Dernière mise à jour : 2026-08-21 (commit
`d094713` sur `main`).

## 1. Ce qui est LIVE (état au 2026-08-21 soir)

- **Modal** : app `aloepri-qwen3-modal` déployée — `POST /generate` (IDs
  permutés, posture stricte : aucune clé sur le serveur), `GET /health`,
  auth Bearer (Secret `aloepri-api-key`). URL :
  `https://mauceri--aloepri-qwen3-modal-serve.modal.run`.
- **Volume Modal** : `aloepri-models` (`/qwen3-8b-obf`, 14 shards, 16 Go).
  Modèle transformé **α_e=0.3, β=8** (réglage « qualité » + défense d'attention Ẑ restaurée le 2026-08-22).
- **Local (cette machine)** : venv `/home/cmauceri/deepseek-harness-ws/venv`
  (torch CPU, transformers 5.15, modal 1.5.4) ; proxy OpenAI-compatible
  (port 8001) et serveur notebook (port 8002) — à redémarrer si besoin
  (§4) ; credentials Modal `~/.modal.toml` (profil `mauceri`), token HF dans
  `~/.cache/huggingface`.

## 2. Artefacts et versions (PIÈGE)

| Artefact | Chemin | Version |
|---|---|---|
| Modèle obfusqué LOCAL | `artifacts/obfuscated_qwen3_8b/` (16 Go) | **α_e=1.0, β=8** (première transform) |
| Modèle sur Volume Modal | `aloepri-models:/qwen3-8b-obf` | **α_e=0.3, β=8** (servi) |
| Clés (identiques partout) | `artifacts/obfuscation_keys.json` | seed 0 — la permutation ne dépend PAS de α_e/β |

**Piège** : le modèle local et le modèle servi ne sont PAS la même version.
Les clés sont les mêmes (même seed/vocabulaire). Toute mesure « sur le modèle
servi » passe par Modal ; toute mesure locale utilise le modèle local
α_e=1.0/β=8 (à re-transformer en local si vous voulez α_e=0.3/β=1).

## 3. Clés et secrets (à ne JAMAIS pousser)

- `artifacts/obfuscation_keys.json` (secret client, 4,8 Mo, SHA-256
  `30f0b58e…`) + `artifacts/obfuscation_keys_modal.json` (copie identique).
- `~/.aloepri-api-key` (clé API de l'endpoint Modal).
- Le Volume Modal `aloepri-keys` est supprimé ; il se recrée VIDE à chaque
  `modal run`/`deploy` (import de `app.py` → `Volume.from_name(create_if_missing=True)`)
  — le supprimer à nouveau (`modal volume delete aloepri-keys -y`) si une
  posture stricte est requise. Un `transform()` y écrirait les clés (par
  design) : les récupérer puis supprimer.

## 4. Commandes de redémarrage

```bash
# proxy OpenAI-compatible local (clés restent locales)
cd ~/deepseek-harness-ws/Secretarius
~/deepseek-harness-ws/venv/bin/python aloepri_modal/openai_proxy.py \
    --keys ~/deepseek-harness-ws/artifacts/obfuscation_keys.json \
    --url https://mauceri--aloepri-qwen3-modal-serve.modal.run \
    --api-key "$(cat ~/.aloepri-api-key)" --port 8001

# serveur notebook (mêmes arguments, --port 8002)
# déploiement / santé
~/deepseek-harness-ws/venv/bin/modal deploy aloepri_modal/app.py
~/deepseek-harness-ws/venv/bin/modal app stop aloepri-qwen3-modal -y   # arrêt
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $(cat ~/.aloepri-api-key)" \
    https://mauceri--aloepri-qwen3-modal-serve.modal.run/health
```

## 5. Prochaines étapes identifiées

1. ~~Restaurer la défense d'attention (β=8)~~ — **FAIT le 2026-08-22** :
   modèle servi en α_e=0.3/β=8, attaque ISA re-mesurée (attn 27,3 % → 9,1 %,
   hidden L1 95,5 % → 90,9 % — cf. `aloepri_poc/RESULTATS_ISA.md`).
2. **Matrices clés P̂/Q̂ (h>0) + chaînage inter-couches** — PROCHAINE GRANDE
   ÉTAPE (choisie le 2026-08-22) : cible TTRSR hidden ≈ 0,82 % (Tableau 4
   AloePri). Design et reprise :
   `docs/superpowers/specs/2026-08-22-aloepri-matrices-cles-design.md`
   (la brique `key_matrix.py` existe et P̂·Q̂=I est vérifié ; reste le
   redimensionnement d+2h + le chaînage §5.4).
3. **Comparaison baseline vs obfusqué** en grandeur nature (l'attaque ISA sur
   la baseline récupérerait les ids CLAIRS — le texte fuirait ; sur
   l'obfusqué elle ne récupère que des ids permutés).
4. **Attaque attention multi-couches** / autre fonction de perte (le canal
   attn diverge avec la méthode actuelle : 27,3 %).
5. **Analyse LoRA** — obfusquer des adaptateurs LoRA et les adjoindre au base
   obfusqué : faisable en principe (toutes les transformations sont linéaires)
   — argumentation et conditions dans `aloepri_poc/CONCLUSION.md` §3 ; à
   valider par un test sur modèle jouet (base obfusquée + adaptateur transformé
   == base originale + adaptateur).
5. Qualité/vitesse sur Qwen3 (mesures `measure_quality.py`/`measure_speed.py`,
   nécessitent une machine avec 16 Go+ de RAM ou un GPU).

## 6. Pièges techniques déjà rencontrés (économise une session)

- `@modal.web_server` + `uvicorn.run` bloquant → la passerelle renvoie 303 ;
  utiliser **`@modal.asgi_app()`** (la fonction RETOURNE l'app FastAPI).
- Valeurs de Secret lues via **`os.environ`** (pas de `.get()` sur l'objet
  Secret dans modal 1.5.4).
- `ephemeral_disk` Modal : minimum **524288 MiB** (512 GiB).
- Le POC est copié dans l'image via `add_local_dir(..., copy=True)` à
  `/pkg/aloepri_poc` — `sys.path.insert(0, "/pkg/aloepri_poc")` ; l'image se
  reconstruit quand les fichiers locaux changent.
- CLI `modal run` : pas d'annotation `list[int]` sur les paramètres (passer
  les ids en CSV).
- `apply_chat_template(tokenize=True)` renvoie un objet `Encoding` selon les
  versions — passer par `tokenize=False` puis `tokenizer(...)`.
- Qwen3-8B est un modèle *thinking* : décodage validé = `enable_thinking=False`
  + greedy + `repetition_penalty=1.05` + blocage du token `<think>` (151667).
- L'attaque ISA sur le vrai modèle : perte **relative** (MSE/variance),
  recuit aligné sur le run, **phase 2** d'affinage, GPU **A100-40GB** (OOM
  sur L4).
