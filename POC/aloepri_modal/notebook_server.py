"""Petit serveur local pour notebook Jupyter : encodage / interrogation /
décodage des prompts vers le modèle obfusqué servi sur Modal.

Le notebook envoie du TEXTE à ce serveur (127.0.0.1) ; lui seul connaît le
tokenizer et les clés. Il permute les IDs, appelle `POST /generate` de Modal
avec des nombres, puis déper mute et décode la réponse. Modal ne voit jamais
de texte ni de clés (posture stricte du POC).

Usage:
    python notebook_server.py \
        --keys /path/to/obfuscation_keys.json \
        --url https://mauceri--aloepri-qwen3-modal-serve.modal.run \
        --api-key "$(cat ~/.aloepri-api-key)" \
        --port 8002

Endpoints :
    GET  /health               → {"status": "ok"}
    POST /ask                  → {"result": str, "usage": {...}}
         body: {"prompt": str, "system": str|None, "max_new_tokens": int}
"""
import argparse
import json
import os
import sys

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from aloepri_poc.client_wrapper import ClientCodec  # noqa: E402

THINK_CLEAR_ID = 151667   # <think> (Qwen3) — bloqué pour réponse directe
REPETITION_PENALTY = 1.05


class AskRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_new_tokens: int = 300


def make_app(modal_url, api_key, codec, tokenizer, model_name):
    app = FastAPI(title="AloePri notebook server (local)")

    def _permutation():
        return codec.permutation

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest):
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        # 1. chat template (non-thinking — régime validé) → IDs clairs
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        clear_ids = tokenizer(templated)["input_ids"]
        # 2. permutation locale (secret côté client)
        permuted = [_permutation()[i] for i in clear_ids]
        # 3. appel Modal /generate : des nombres, rien d'autre
        payload = {
            "input_ids": permuted,
            "max_new_tokens": req.max_new_tokens,
            "repetition_penalty": REPETITION_PENALTY,
            # <think> interdit, exprimé dans l'espace permuté (opaque au
            # serveur)
            "bad_words_ids": [[_permutation()[THINK_CLEAR_ID]]],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(modal_url.rstrip("/") + "/generate",
                             json=payload, headers=headers, timeout=600)
        if resp.status_code != 200:
            raise HTTPException(status_code=502,
                                detail=f"Modal /generate: {resp.status_code} "
                                       f"{resp.text[:300]}")
        out_ids = resp.json()["output_ids"]
        # 4. dépermutation + décodage côté client
        completion = out_ids[len(permuted):]
        result = tokenizer.decode(
            [codec.unpermute[i] for i in completion],
            skip_special_tokens=True).strip()
        return {
            "result": result,
            "usage": {
                "prompt_tokens": len(clear_ids),
                "completion_tokens": len(completion),
                "total_tokens": len(clear_ids) + len(completion),
            },
        }

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", required=True,
                    help="obfuscation_keys.json (secret client, jamais envoyé)")
    ap.add_argument("--url", required=True, help="URL du endpoint Modal /generate")
    ap.add_argument("--api-key", default=os.environ.get("ALOEPRI_API_KEY"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    with open(args.keys) as f:
        keys = json.load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = ClientCodec(keys["vocab_permutation"], keys["vocab_unpermute"],
                        tokenizer)
    app = make_app(args.url, args.api_key, codec, tokenizer,
                   "qwen3-8b-obf")
    print(f"[notebook-server] http://127.0.0.1:{args.port}/ask "
          f"(clés locales ; Modal ne reçoit que des IDs permutés)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
