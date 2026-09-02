"""Client chiffré de démonstration — LLM obfusqué AloePri (h>0) servi sur Modal.

La confidentialité repose sur la permutation de vocabulaire Π, qui reste
EXCLUSIVEMENT côté client : ce script tokenize avec le tokenizer public
Qwen3, permute les ids avec la clé locale (`obfuscation_keys.json`), envoie
les ids permutés au service, puis dépermute et décode la réponse. Le serveur
(et le transport) ne voient jamais de texte en clair ni la clé.

Usage (démo) :
    .venv/bin/python tools/client_demo.py \\
        --url https://<workspace>--obfuscator-aloepri-serve.modal.run \\
        --keys artifacts/obfuscation_keys.json \\
        --api-key "$(cat ~/.aloepri-api-key)" \\
        "Quelle est la capitale de la France ?"

Sortie (mode verbose par défaut) : ids clairs → ids permutés envoyés → ids
permutés reçus → ids clairs → texte. C'est le cœur démontrable : le transport
ne contient que des nombres sans signification pour qui n'a pas Π.
"""
import argparse
import json
import os
import sys

import requests
from transformers import AutoTokenizer


class ClientCodec:
    """Encode/décode côté client : Π à l'envoi, Π⁻¹ à la réception.

    Les clés du JSON sont des chaînes : coercition en int une seule fois ici.
    """

    def __init__(self, permutation, unpermute, tokenizer):
        self.permutation = {int(k): int(v) for k, v in permutation.items()}
        self.unpermute = {int(k): int(v) for k, v in unpermute.items()}
        self.tokenizer = tokenizer

    def encode(self, text):
        return [self.permutation[i] for i in self.tokenizer.encode(text)]

    def decode(self, permuted_ids):
        clear = [self.unpermute[i] for i in permuted_ids]
        return self.tokenizer.decode(clear)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True,
                    help="URL de base du service Modal (ex. https://…modal.run)")
    ap.add_argument("--keys", required=True,
                    help="obfuscation_keys.json — secret client, JAMAIS envoyé")
    ap.add_argument("--api-key", default=None,
                    help="Bearer attendu par le service (fail-closed)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                    help="tokenizer public (tokenize/detokenize)")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--quiet", action="store_true",
                    help="n'affiche que le texte déchiffré")
    ap.add_argument("prompt")
    args = ap.parse_args()

    with open(args.keys) as f:
        keys = json.load(f)
    codec = ClientCodec(keys["vocab_permutation"], keys["vocab_unpermute"],
                        AutoTokenizer.from_pretrained(args.tokenizer))

    clear_ids = codec.tokenizer.encode(args.prompt)
    permuted_in = codec.encode(args.prompt)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    # health (fail-closed : 401 sans clé attendue côté serveur)
    r = requests.get(args.url.rstrip("/") + "/health", headers=headers,
                     timeout=30)
    r.raise_for_status()

    payload = {"input_ids": permuted_in,
               "max_new_tokens": args.max_new_tokens}
    r = requests.post(args.url.rstrip("/") + "/generate", json=payload,
                      headers=headers, timeout=300)
    r.raise_for_status()
    permuted_out = r.json()["output_ids"][len(permuted_in):]

    if not args.quiet:
        print("ids clairs (prompt)     :", clear_ids)
        print("ids PERMUTÉS envoyés    :", permuted_in)
        print("ids PERMUTÉS reçus (gen):", permuted_out)
        print("— le transport ne contient que ces nombres —")
    print(codec.decode(permuted_out))


if __name__ == "__main__":
    main()
