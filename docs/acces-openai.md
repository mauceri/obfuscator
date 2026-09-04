# Accès au Qwen3-14B obfusqué servi (protocole OpenAI)

Le modèle obfusqué `qwen3-14b-h128-a1-h02` est servi en **serverless sur
Modal** (A100-40GB, scale-to-zero). L'accès « protocole OpenAI » passe par un
**proxy local** qui détient la clé de permutation : Modal ne reçoit jamais de
texte ni de clé.

## Architecture

```
Client OpenAI (curl, openai-py, Wiki_LM…)
        │  texte (messages)               protocole OpenAI /v1/chat/completions
        ▼
PROXY LOCAL (tools/openai_proxy.py)  ←── clé Π + tokenizer, SUR VOTRE MACHINE
        │  chat template non-thinking → tokenize → PERMUTE les ids
        │  (des nombres, rien d'autre)
        ▼
SERVICE Modal (serverless, A100-40GB)
        │  POST /generate — ids permutés → ids permutés
        ▼
PROXY LOCAL : dépermute + décode → texte de la réponse
```

## 1. Le service serverless (déjà déployé)

- URL : `https://mauceri--obfuscator-aloepri-serve.modal.run`
- Modèle servi : `qwen3-14b-h128-a1-h02` (h>0, α_e=1,0/α_h=0,2, hidden 5376)
- Endpoint brut : `POST /generate` (ids permutés) + `GET /health`
- Redéployer après un changement de code : `~/modal-venv/bin/modal deploy modal_app.py`

Vérification (sans le proxy) :

```bash
curl -H "Authorization: Bearer $(cat ~/.aloepri-api-key)" \
  https://mauceri--obfuscator-aloepri-serve.modal.run/health
# → {"status":"ok"}   (401 sans Bearer — fail-closed)
```

## 2. Lancer le proxy local (clé + tokenizer sur votre machine)

```bash
cd <repo-obfuscator>
~/Secretarius/Wiki_LM/.venv/bin/python tools/openai_proxy.py \
    --keys artifacts/obfuscation_keys.json \
    --url https://mauceri--obfuscator-aloepri-serve.modal.run \
    --api-key "$(cat ~/.aloepri-api-key)" \
    --port 8001
```

Le proxy applique le **chat template non-thinking** (`enable_thinking=False`),
tokenize + permute localement, envoie les ids permutés, dépermeute la réponse.

## 3. Y accéder comme à une API OpenAI

**curl :**

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-14b-h128-a1-h02",
       "messages": [{"role": "user",
                     "content": "Quelle est la capitale de la France ?"}],
       "max_tokens": 100}'
```

**Client Python `openai` :**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="local")
r = client.chat.completions.create(
    model="qwen3-14b-h128-a1-h02",
    messages=[{"role": "user", "content": "Résume ce texte en 3 phrases : …"}],
    max_tokens=200)
print(r.choices[0].message.content)
```

**Wiki_LM** (backend OpenAI déjà câblé dans `tools/llm.py`) — dans
`~/Secretarius/Wiki_LM/.env` :

```
WIKI_LLM_BACKEND=openai
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_MODEL=qwen3-14b-h128-a1-h02
OPENAI_API_KEY=local            # le proxy n'exige pas de clé côté client
```

## 4. Sécurité et posture

- Modal ne voit que des **ids permutés** (nombres) — jamais de texte, jamais
  la clé, jamais le tokenizer ;
- la clé Π (`artifacts/obfuscation_keys.json`, seed 0 expérimentale) et le
  tokenizer restent **sur votre machine** ;
- le proxy est en écoute sur `127.0.0.1` uniquement — ne pas l'exposer ;
- le `stop_token_id` (id permuté de `<|im_end|>`) est fourni par le proxy :
  sans lui, le générateur ne s'arrête jamais (corrigé — l'eos du config est
  l'id clair, jamais émis par le modèle obfusqué).

## 5. Coût

Serverless A100-40GB (~1,5-2 $/h, scale-to-zero) : facturé au temps GPU
réellement allumé + cold start (~1-2 min de chargement des 30 Go après une
période d'inactivité). Pour un usage par lots modéré (résumés Wiki_LM), le
coût mensuel est de l'ordre de quelques dollars.

## 6. Limitations connues

- **Pas de streaming** (réponse complète renvoyée — greedy, `temperature`
  accepté mais ignoré) ;
- un seul tour de génération par appel (le proxy gère le multi-tour en
  réenvoyant l'historique) ;
- contexte max : fenêtre du modèle (le 14B supporte 131k, mais le service
  n'a pas de limite explicite côté serveur — borné par la VRAM KV cache).
