"""Proxy OpenAI-compatible LOCAL vers le modèle obfusqué servi sur Modal.

Posture STRICTE : les clés de permutation et le tokenizer restent sur cette
machine. Le proxy reçoit des messages (format OpenAI), applique le chat
template NON-THINKING, tokenize + permute localement, appelle
`POST /generate` de Modal avec des IDs PERMUTÉS (de simples nombres), puis
dépermute et décode la réponse.

Modal ne voit jamais : du texte, le tokenizer, ni les clés.

Usage:
    python tools/openai_proxy.py \\
        --keys artifacts/obfuscation_keys.json \\
        --url https://mauceri--obfuscator-aloepri-serve.modal.run \\
        --api-key "$(cat ~/.aloepri-api-key)" \\
        --model-name qwen3-14b-h128-a1-h02 \\
        --port 8001

Puis, comme un vrai endpoint OpenAI :
    curl http://127.0.0.1:8001/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model": "qwen3-14b-h128-a1-h02",
             "messages": [{"role": "user",
                           "content": "Quelle est la capitale de la France ?"}],
             "max_tokens": 300}'

Le chat template `enable_thinking=False` est appliqué LOCALEMENT (le Qwen3
chat « pense » sinon — vérifié : continuation directe → dérive en questions).

Dépendances : requests, fastapi, uvicorn, transformers (venv scientifique).
"""
import argparse
import json
import os
import time
import uuid

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

THINK_CLEAR_ID = 151667   # <think> (Qwen3) — id CLAIR, permuté avant l'envoi
REPETITION_PENALTY = 1.05


class ClientCodec:
    """Permutation du vocabulaire — le secret reste sur cette machine."""

    def __init__(self, permutation, unpermute, tokenizer):
        self.permutation = {int(k): int(v) for k, v in permutation.items()}
        self.unpermute = {int(k): int(v) for k, v in unpermute.items()}
        self.tokenizer = tokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 300
    temperature: float | None = None   # accepté, ignoré (greedy)
    stream: bool = False


def make_app(modal_url, api_key, codec, tokenizer, model_name):
    app = FastAPI(title=f"AloePri {model_name} (proxy local)")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model",
             "created": int(time.time()), "owned_by": "mauceri"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "proxy": True}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(status_code=400,
                                detail="streaming non supporté (greedy)")
        # 1. chat template non-thinking → texte → IDs CLAIRS
        templated = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        clear_ids = tokenizer(templated)["input_ids"]
        # 2. permutation locale (secret côté client)
        permuted = [codec.permutation[i] for i in clear_ids]
        # 3. appel Modal /generate : des nombres, rien d'autre
        payload = {
            "input_ids": permuted,
            "max_new_tokens": req.max_tokens,
            "repetition_penalty": REPETITION_PENALTY,
            # <think> interdit, exprimé dans l'espace permuté (opaque au
            # serveur : un simple id à ne pas émettre)
            "bad_words_ids": [[codec.permutation[THINK_CLEAR_ID]]],
        }
        # token d'arrêt = <|im_end|> (151645) exprimé dans l'espace permuté
        payload["stop_token_id"] = codec.permutation[151645]
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", required=True,
                    help="obfuscation_keys.json (secret client, jamais envoyé)")
    ap.add_argument("--url", required=True, help="URL Modal /generate")
    ap.add_argument("--api-key", default=os.environ.get("ALOEPRI_API_KEY"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--model-name", default="qwen3-14b-h128-a1-h02")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    with open(args.keys) as f:
        keys = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = ClientCodec(keys["vocab_permutation"],
                        keys["vocab_unpermute"], tokenizer)
    app = make_app(args.url, args.api_key, codec, tokenizer, args.model_name)
    print(f"[proxy] OpenAI-compatible local : "
          f"http://127.0.0.1:{args.port}/v1/chat/completions")
    print(f"[proxy] modèle '{args.model_name}' — clés et tokenizer LOCAUX, "
          "Modal ne reçoit que des IDs permutés")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
