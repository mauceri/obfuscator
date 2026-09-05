"""Client du modèle obfusqué AloePri servi sur Modal.

Le serveur ne connaît que des IDs de tokens PERMUTÉS : ce client tokenize
avec le tokenizer public Qwen3, permute les IDs avec la clé locale
(`obfuscation_keys.json`), envoie, puis dépermute et décode la réponse.

Usage:
    python client.py --url https://mauceri--aloepri-qwen3-modal-serve.modal.run \\
        --keys ./obfuscation_keys.json \\
        --api-key "$ALOEPRI_API_KEY" \\
        "Quelle est la capitale de la France ?"

Sans `--api-key`, l'endpoint n'est protégé par rien (Secret Modal absent) —
ne pas déployer sans clé sur un endpoint exposé publiquement.
"""
import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from aloepri_poc.client_wrapper import ClientCodec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True,
                    help="URL de base du endpoint Modal (ex. https://…modal.run)")
    ap.add_argument("--keys", required=True,
                    help="obfuscation_keys.json (secret client, ne jamais "
                         "l'envoyer au serveur)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                    help="tokenizer public utilisé pour tokenize/detokenize")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("prompt")
    args = ap.parse_args()

    with open(args.keys) as f:
        keys = json.load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = ClientCodec(keys["vocab_permutation"], keys["vocab_unpermute"],
                        tokenizer)

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    permuted_ids = codec.encode(args.prompt)
    print(f"[client] {len(permuted_ids)} tokens → envoi des IDs permutés à "
          f"{args.url} (greedy, {args.max_new_tokens} tokens max)")
    resp = requests.post(
        args.url.rstrip("/") + "/generate",
        json={"input_ids": permuted_ids,
              "max_new_tokens": args.max_new_tokens},
        headers=headers, timeout=600,
    )
    resp.raise_for_status()
    output_permuted = resp.json()["output_ids"]
    print("[client] sortie décodée :")
    print(codec.decode(output_permuted))


if __name__ == "__main__":
    main()
