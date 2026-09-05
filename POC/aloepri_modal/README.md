# AloePri sur Modal — Qwen3-8B obfusqué en service serverless

Pipeline complet : transformation AloePri du modèle [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
(permutation de vocabulaire + bruit embedding + attention/FFN obfusqués,
d'après le POC `aloepri_poc/` — méthode du papier
[AloePri, arXiv 2603.01499](https://arxiv.org/pdf/2603.01499)) puis service
HTTP sur un GPU Modal.

**Posture de sécurité (strict)** : le serveur ne voit JAMAIS les clés. Il ne
charge aucun tokenizer, ne reçoit que des **IDs de tokens permutés** (des
nombres) et ne renvoie que des IDs permutés. La permutation du vocabulaire
(`obfuscation_keys.json`) et le tokenizer restent exclusivement côté client —
voir le codec (`aloepri_poc/client_wrapper.py`) et le proxy OpenAI-compatible
local (§6bis). Le Volume `aloepri-keys` n'existe plus sur Modal (supprimé
après récupération des clés).

## Prérequis

- CLI Modal (sur la machine de déploiement, ex. sanroque) :
  ```bash
  python3 -m venv ~/modal-venv && ~/modal-venv/bin/pip install -U modal
  ~/modal-venv/bin/modal setup        # auth navigateur (ou: modal token new)
  ```
- Le modèle obfusqué + les clés, produits par
  `aloepri_poc/transform_streaming.py` (cf. §1), OU reproduits sur Modal (§2).
- `aloepri_poc/` et `aloepri_modal/` à jour dans le dépôt `Secretarius`.

## 1. Produire le modèle obfusqué (recommandé : en local, machine ≥ 6 Go RAM)

`transform_streaming.py` transforme Qwen3-8B shard par shard (pic ≈ 5 Go,
16 Go de téléchargement, ~30-60 min sur CPU) et écrit les clés SÉPARÉMENT :

```bash
cd ~/Secretarius/aloepri_poc
python3 check_arch.py --model Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B
#   → doit afficher « toutes les hypothèses sont satisfaites »
#     (q_norm/k_norm détectés → rope_scaling=off automatique)

python3 transform_streaming.py --model Qwen/Qwen3-8B \
    --output /path/to/obfuscated_qwen3_8b \
    --keys /path/to/obfuscation_keys.json --seed 0
```

Vérifier sans charger les 16 Go (`verify_transform.py` recalcule des
échantillons de lignes/couches et les compare bit-à-bit) :

```bash
python3 verify_transform.py --model-dir /path/to/obfuscated_qwen3_8b \
    --keys /path/to/obfuscation_keys.json --source Qwen/Qwen3-8B
```

**Les clés sont le secret.** Les garder côté client, hors du dépôt et hors de
Modal ; ne JAMAIS les copier dans `aloepri-models`.

## 2. (Alternative) Reproduire la transformation sur Modal

```bash
~/modal-venv/bin/modal run aloepri_modal/app.py::transform \
    --seed 0 --alpha-e 1.0
```

Écrit le modèle sur le Volume `aloepri-models` (`/qwen3-8b-obf`) et les clés
sur `aloepri-keys`. Récupérer ensuite les clés (cf. §3) puis SUPPRIMER le
Volume des clés après téléchargement :

```bash
~/modal-venv/bin/modal volume get aloepri-keys /obfuscation_keys.json ./obfuscation_keys.json
~/modal-venv/bin/modal volume delete aloepri-keys
```

## 3. Volumes + upload du modèle

```bash
~/modal-venv/bin/modal volume create aloepri-models        # une fois
~/modal-venv/bin/modal volume put aloepri-models \
    /path/to/obfuscated_qwen3_8b /qwen3-8b-obf             # ~16 Go
```

Vérifier : `~/modal-venv/bin/modal volume ls aloepri-models /qwen3-8b-obf`.

## 4. (Optionnel) Protéger l'endpoint par clé API

```bash
openssl rand -hex 32 > ~/.aloepri-api-key && chmod 600 ~/.aloepri-api-key
~/modal-venv/bin/modal secret create aloepri-api-key \
  ALOEPRI_API_KEY="$(cat ~/.aloepri-api-key)"
```

Sans ce Secret, `serve()` répond sans authentification.

## 5. Déployer

```bash
~/modal-venv/bin/modal deploy aloepri_modal/app.py
# imprime l'URL, ex. https://mauceri--aloepri-qwen3-modal-serve.modal.run
```

Comportement serverless : conteneur éteint après ~5 min sans requête
(scale-to-zero). La requête suivante subit le cold start (boot + chargement
des 16 Go depuis le Volume) : l'endpoint peut répondre `503` pendant ~1-3 min
— réessayer. Réveil avant usage :

```bash
URL=https://mauceri--aloepri-qwen3-modal-serve.modal.run
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health")" = "200" ]; do sleep 5; done
```

## 6. Round-trip de bout en bout (critère de succès n°1)

```bash
python3 aloepri_modal/client.py \
    --url "$URL" \
    --keys /path/to/obfuscation_keys.json \
    --api-key "$(cat ~/.aloepri-api-key)" \
    "Quelle est la capitale de la France ?"
```

Vérifier MANUELLEMENT que la sortie est un texte français cohérent (le POC
Qwen2.5 mesurait +19 % de perplexité moyenne — cf. `aloepri_poc/RESULTATS.md` ;
le round-trip reste qualitativement correct).

## 6bis. Interface OpenAI-compatible — mais côté CLIENT (proxy local)

Modal ne sert que `/generate` (IDs permutés). Pour obtenir une interface
`/v1/chat/completions` SANS mettre les clés sur le serveur, on lance un proxy
local qui fait la permutation (tokenizer + clés restent sur votre machine) :

```bash
python3 aloepri_modal/openai_proxy.py \
    --keys /path/to/obfuscation_keys.json \
    --url "$URL" \
    --api-key "$(cat ~/.aloepri-api-key)" \
    --port 8001
# → http://127.0.0.1:8001/v1/chat/completions
```

Puis, comme un vrai endpoint OpenAI :

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "qwen3-8b-obf",
         "messages": [{"role": "user", "content": "Quelle est la capitale de la France ?"}],
         "max_tokens": 300}'
# → {"object": "chat.completion", "choices": [{"message": {"content": "La capitale de la France est Paris. 😊"}}], "usage": {...}}
```

Ce que le proxy envoie à Modal, c'est uniquement `{"input_ids": […], …}` —
des nombres. Les paramètres de décodage (repetition_penalty, bad_words_ids)
sont des nombres opaques pour le serveur.

**Réglages de décodage** (validés en grandeur nature, cf. `RESULTATS_QWEN3.md`) :
le modèle servi est transformé avec **`--alpha-e 0.3 --beta 1`** (perturbation
minimale : bruit d'embedding réduit + attention exacte sans approximation Ẑ)
et le proxy utilise `enable_thinking=False` (réponse directe), greedy,
`repetition_penalty=1.05` et bloque le token `<think>` — Qwen3-8B (modèle
*thinking*) a tendance à ouvrir des traces de raisonnement qui bouclent
sinon. Les clés sont inchangées (même seed, même vocabulaire : la permutation
ne dépend pas de α_e/β).

> Historique : un endpoint serveur `/analyze` (texte→texte, clés montées) a
> été testé puis RETIRÉ : il plaçait les moyens de désobfuscation sur le
> serveur. La forme retenue est le proxy local ci-dessus — strict.

## 7. Mesures qualité / vitesse (optionnel)

Reprendre les scripts du POC sur le modèle obfusqué local (les clés ne sont
nécessaires QUE pour ces mesures, jamais pour le serveur) :

```bash
cd ~/Secretarius/aloepri_poc
python3 measure_quality.py --baseline Qwen/Qwen3-8B \
    --obfuscated /path/to/obfuscated_qwen3_8b --keys /path/to/obfuscation_keys.json
python3 measure_speed.py --baseline Qwen/Qwen3-8B \
    --obfuscated /path/to/obfuscated_qwen3_8b --keys /path/to/obfuscation_keys.json
```

## Coût

- Service : GPU L4 ≈ 0,80 $/h, facturé au temps d'allumage GPU uniquement
  (scale-to-zero). A100-40GB (~1,5-2 $/h) si contextes très longs.
- Transformation locale : gratuite (CPU). Sur Modal : quelques dizaines de
  minutes de conteneur CPU.

## Résolution de problèmes

| Symptôme | Cause / remède |
|---|---|
| `503` au premier appel | cold start (chargement 16 Go) — attendre, réessayer |
| `401 unauthorized` | Secret `aloepri-api-key` absent ou `--api-key` erroné |
| Sortie incohérente | clés non appariées au modèle (seed/paramètres différents) — refaire §1 ou §2 sans changer les paramètres |
| GPU OOM sur très long contexte | passer `GPU_SERVE = "A100-40GB"` dans `app.py` |
| `403` sur volume put | le Volume `aloepri-models` n'existe pas — `modal volume create` |
| Qwen3-8B introuvable | modèle public HF, aucun token requis |

## Sécurité (à relire)

1. `obfuscation_keys.json` = secret client. Pas dans le dépôt
   (`aloepri_poc/.gitignore`), pas sur Modal (Volume `aloepri-keys` supprimé),
   pas dans les logs Modal (le SHA-256 des clés est OK ; le fichier non).
2. Le serveur (`/generate`) ne reçoit que des IDs permutés — aucun moyen de
   désobfusquer côté serveur. Le proxy local (§6bis) porte le tokenizer + les
   clés sur votre machine et n'envoie que des nombres à Modal.
3. Le modèle obfusqué SANS les clés ne permet pas de décoder les échanges —
   c'est le point du POC. Un serveur compromis expose les poids obfusqués et
   les IDs permutés, pas la permutation elle-même.
4. Limites connues du POC (inchangées depuis `aloepri_poc/RESULTATS.md`) :
   pas de matrices clés P̂/Q̂ (h=0), pas de chaînage inter-couches, bruit
   d'embedding non compensé (α_e — levier qualité principal : 0,5 au lieu de
   1,0 améliore la perplexité de ~19 % à ~14 % sur Qwen2.5).
