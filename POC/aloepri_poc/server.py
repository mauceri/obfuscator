"""Serveur HTTP minimal servant le modèle obfusqué (ou baseline) via transformers.

Le serveur travaille UNIQUEMENT au niveau des IDs de tokens : il ne charge
aucun tokenizer. Le tokenizer n'est pas sauvegardé avec le modèle obfusqué —
la table d'embedding est réindexée par la permutation de vocabulaire, mais le
vocabulaire textuel (le tokenizer public Qwen) reste un secret côté client,
avec les clés (`obfuscation_keys.json`, cf. `model_transform.py`). C'est
`client_wrapper.ClientCodec` qui tokenize+permute avant l'envoi et
dépermute+détokenize à la réception ; un `AutoTokenizer.from_pretrained(model_dir)`
ici planterait de toute façon, le répertoire du modèle obfusqué ne contenant
aucun fichier de tokenizer (décision de design, pas un oubli).
"""
import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM

app = FastAPI()
_model = None


class GenerateRequest(BaseModel):
    input_ids: list[int]
    max_new_tokens: int = 100


class GenerateResponse(BaseModel):
    output_ids: list[int]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    input_tensor = torch.tensor([req.input_ids], device=_model.device)
    output = _model.generate(input_tensor, max_new_tokens=req.max_new_tokens, do_sample=False)
    return GenerateResponse(output_ids=output[0].tolist())


def load(model_dir):
    global _model
    # dtype= (pas torch_dtype=, déprécié dans transformers>=4.56) : le
    # checkpoint obfusqué a été sauvegardé en bfloat16 par `model_transform.py`
    # (Û_vo orthogonale rend ce dtype numériquement sûr, cf. sa docstring) —
    # charger dans un autre dtype romprait cette correspondance.
    _model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load(args.model_dir)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
