# Unités systemd des proxies OpenAI locaux (AloePri)

Les deux proxies qui traduisent le protocole OpenAI vers les services
obfusqués Modal (`/generate`, ids permutés). La clé Π et le tokenizer
restent **locaux** ; Modal ne voit que des nombres.

| Unité | Modèle servi | Endpoint Modal | Port local |
|---|---|---|---|
| `obfuscator-proxy.service` | Qwen3-14B obfusqué | `...-aloepri-serve.modal.run` | **8001** |
| `obfuscator-proxy-8b.service` | Qwen3-8B obfusqué | `...-aloepri-serve-8b.modal.run` | **8002** |

## Installation

```bash
# 1. copier les unités
cp systemd/obfuscator-proxy.service systemd/obfuscator-proxy-8b.service \
   ~/.config/systemd/user/

# 2. fichier d'environnement (clé du service Modal — fail-closed)
mkdir -p ~/.config/obfuscator
printf 'ALOEPRI_API_KEY=%s\n' "$(cat ~/.aloepri-api-key)" \
    > ~/.config/obfuscator/proxy.env
chmod 600 ~/.config/obfuscator/proxy.env

# 3. adapter les chemins ExecStart si nécessaire : %h = home ; le venv
#    scientifique (~/Secretarius/Wiki_LM/.venv) peut être remplacé par
#    <repo>/.venv ; le repo est attendu dans %h/obfuscator.

# 4. activer
systemctl --user daemon-reload
systemctl --user enable --now obfuscator-proxy obfuscator-proxy-8b
systemctl --user status obfuscator-proxy
```

## Exposition Tailscale (accès depuis le tailnet)

```bash
tailscale serve --bg --https=9443 http://127.0.0.1:8001   # 14B
tailscale serve --bg --https=9444 http://127.0.0.1:8002   # 8B
```
→ `https://<machine>.<tailnet>.ts.net:9443/v1` (protocole OpenAI).

## Vérification

```bash
curl http://127.0.0.1:8001/health                      # {"status":"ok","proxy":true}
curl http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b-h128-a1-h02","messages":[{"role":"user","content":"2+2 ?"}],"max_tokens":20}'
```

## Notes

- Le proxy exige les dépendances : `fastapi`, `uvicorn`, `requests`,
  `transformers` (venv scientifique) — sinon `pip install` dans le venv.
- Après une rotation de clé (`notebooks/rotation_pi.ipynb`) : mettre à jour
  `artifacts/obfuscation_keys.json` puis
  `systemctl --user restart obfuscator-proxy obfuscator-proxy-8b`.
- La config Tailscale `serve` persiste (réappliquée au démarrage du daemon).
