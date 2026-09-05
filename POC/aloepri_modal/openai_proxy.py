"""Proxy OpenAI-compatible LOCAL vers le modèle obfusqué servi sur Modal.

Posture STRICTE : les clés de permutation et le tokenizer restent sur cette
machine. Le proxy reçoit des messages (format OpenAI), tokenize + permute
localement, appelle `POST /generate` de Modal avec des IDs PERMUTÉS (de
simples nombres), puis déper mute et décode la réponse.

Modal ne voit jamais : du texte, le tokenizer, ni les clés.

Usage:
    python openai_proxy.py \
        --keys /path/to/obfuscation_keys.json \
        --url https://mauceri--aloepri-qwen3-modal-serve.modal.run \
        --api-key "$(cat ~/.aloepri-api-key)" \
        --port 8001

Puis, comme un vrai endpoint OpenAI :
    curl http://127.0.0.1:8001/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model": "qwen3-8b-obf",
             "messages": [{"role": "user", "content": "Quelle est la capitale de la France ?"}],
             "max_tokens": 300}'

Réglages de décodage (validés en grandeur nature, cf. RESULTATS_QWEN3.md) :
chat template non-thinking, greedy, repetition_penalty=1.05, blocage du
token <think> — le serveur reçoit ces réglages comme des nombres opaques
(repetition_penalty, bad_words_ids), sans rien révéler de leur sens.
"""
import argparse
import json
import os
import sys
import time
import uuid

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from aloepri_poc.client_wrapper import ClientCodec  # noqa: E402

THINK_CLEAR_ID = 151667   # <think> (Qwen3)
REPETITION_PENALTY = 1.05


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 300
    temperature: float | None = None  # accepté, ignoré (greedy validé)
    stream: bool = False


def build_codec(keys_path, tokenizer_name):
    with open(keys_path) as f:
        keys = json.load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    codec = ClientCodec(keys["vocab_permutation"], keys["vocab_unpermute"],
                        tokenizer)
    return codec, tokenizer


def make_app(modal_url, api_key, codec, tokenizer, model_name):
    app = FastAPI(title="AloePri Qwen3-8B (proxy local)")

    def _permutation():
        return codec.permutation

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": int(time.time()),
             "owned_by": "mauceri"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(status_code=400,
                                detail="streaming non supporté (greedy)")
        # 1. chat template (non-thinking — régime validé) → IDs clairs
        templated = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        clear_ids = tokenizer(templated)["input_ids"]
        # 2. permutation locale (secret côté client)
        permuted = [_permutation()[i] for i in clear_ids]
        # 3. appel Modal /generate : des nombres, rien d'autre
        payload = {
            "input_ids": permuted,
            "max_new_tokens": req.max_tokens,
            "repetition_penalty": REPETITION_PENALTY,
            # <think> interdit, exprimé dans l'espace permuté (opaque au
            # serveur : un simple id à ne pas émettre)
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
        content = tokenizer.decode(
            [codec.unpermute[i] for i in completion],
            skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
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
    ap.add_argument("--model-name", default="qwen3-8b-obf")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    codec, tokenizer = build_codec(args.keys, args.tokenizer)
    app = make_app(args.url, args.api_key, codec, tokenizer, args.model_name)
    print(f"[proxy] OpenAI-compatible local sur http://127.0.0.1:{args.port}"
          f"/v1/chat/completions (modèle '{args.model_name}')")
    print(f"[proxy] les clés restent locales ; Modal ne reçoit que des IDs "
          "permutés")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
