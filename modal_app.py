"""App Modal « obfuscator-aloepri » : transformation, vérification, service et
attaques ISA du modèle obfusqué (package `aloepri`, port T1-T2).

Cinq fonctions appelées par le notebook (T5-T7) :
- `transform()` : reproduit la transformation AloePri sur le modèle source
  (Qwen/Qwen3-8B par défaut) dans un conteneur Modal (streaming mémoire-léger,
  cf. `aloepri/transform_streaming.py`), écrit le modèle obfusqué sur le
  Volume `obfuscator-models` et les clés sur le Volume `obfuscator-keys`.
  Variantes d'arc d'attaques : `obfuscate_attention=False` /
  `apply_permutation=False` (défauts `True`/`True` — comportement historique).
- `verify()` : vérification bit-à-bit par échantillons (lignes embed/head +
  couches complètes) entre le modèle source (cache HF) et le modèle obfusqué
  du Volume, clés régénérées par seed (aucun Volume clés monté).
- `serve()` : serveur HTTP qui charge le modèle obfusqué depuis
  `obfuscator-models` et sert `POST /generate` (IDs de tokens PERMUTÉS) +
  `GET /health`. Il ne voit JAMAIS les clés : le client permute avant l'envoi
  et dépermute à la réception.
- `isa_attack()` : démonstration ISA par gradient sur le modèle obfusqué du
  Volume (ou un modèle HF clair via `model_ref="hf:..."`) — l'attaquant
  (= opérateur du serveur) n'a QUE les poids obfusqués, aucune clé.
- `diag()` : état des Volumes `obfuscator-models` / `obfuscator-keys`.

Posture de sécurité : le Volume `obfuscator-keys` n'est monté QUE par
`transform()` — jamais par `serve()`/`isa_attack()`/`verify()` (la clé de
permutation reste côté client, cf. README).

Déploiement : voir README.md. GPU de service : L4 (24 Go VRAM) — le modèle
pèse ~16 Go en bf16 ; passer à A100-40GB pour de très longs contextes.
"""
import os
import sys

import modal

MODELS_DIR = "/models"
KEYS_DIR = "/keys"
MODEL_VOL = "obfuscator-models"
KEYS_VOL = "obfuscator-keys"
SRC_MODEL = "Qwen/Qwen3-8B"          # source de la transformation
MODEL_SUBDIR = "qwen3-8b-obf"         # sous-répertoire sur le Volume service
KEYS_FILENAME = "obfuscation_keys.json"
GPU_SERVE = "L4"                       # 24 Go VRAM ; A100-40GB si contexte long
TRANSFORM_MEMORY = 12288               # MiB garantis pour transform()/verify()
# Disque éphémère Modal : minimum 512 GiB (524288 MiB), jusqu'à 3 TiB. Le
# cache HF (16 Go) + la sortie (16 Go) tiennent largement dans 512 GiB.
TRANSFORM_EPHEMERAL_DISK = 524288

_ALOEPRI_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aloepri")

# Image du transform : le package aloepri est copié DANS l'image au build
# (pas de Mount runtime), puis importé depuis /pkg/aloepri. scipy requis par
# aloepri/key_matrix.py (scipy.linalg.null_space).
TRANSFORM_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.51", "numpy", "scipy",
        "safetensors", "huggingface_hub",
    )
    .add_local_dir(_ALOEPRI_DIR, "/pkg/aloepri", copy=True)
)
# Image du service : transformers + serveur web, sans le package aloepri
# (inutile ici : le serveur ne permute/dépermute rien).
SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.51", "fastapi", "uvicorn",
        "safetensors",
    )
)

models_vol = modal.Volume.from_name(MODEL_VOL, create_if_missing=True)
keys_vol = modal.Volume.from_name(KEYS_VOL, create_if_missing=True)

try:
    API_SECRET = modal.Secret.from_name("aloepri-api-key")
except modal.exception.NotFoundError:  # pragma: no cover
    API_SECRET = None

app = modal.App("obfuscator-aloepri")


@app.function(
    image=TRANSFORM_IMAGE,
    memory=TRANSFORM_MEMORY,
    ephemeral_disk=TRANSFORM_EPHEMERAL_DISK,
    volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
    timeout=3600,
    scaledown_window=300,
)
def transform(
    seed: int = 0,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    beta: int = 8,
    zeta: float = 1e3,
    rope_scaling: str = "auto",
    model_name: str = SRC_MODEL,
    out_subdir: str = MODEL_SUBDIR,
    obfuscate_attention: bool = True,
    apply_permutation: bool = True,
):
    """Reproduit la transformation AloePri sur le modèle source.

    Sortie : modèle obfusqué sous {MODELS_DIR}/{out_subdir} sur le Volume
    `obfuscator-models` ; clés sous {KEYS_DIR}/{KEYS_FILENAME} sur le Volume
    `obfuscator-keys` (jamais monté par `serve`/`isa_attack`/`verify`).
    Retourne le chemin des clés et leur empreinte SHA-256 — à comparer côté
    client après téléchargement.

    Variantes d'arc d'attaques (T2) : `obfuscate_attention=False` laisse les
    poids d'attention intacts ; `apply_permutation=False` garde les tables
    embed/head dans l'ordre clair (bruit α conservé) et ne remappe pas les
    token_ids spéciaux.
    """
    sys.path.insert(0, "/pkg/aloepri")
    from aloepri.transform_streaming import transform_streaming

    out_dir = os.path.join(MODELS_DIR, out_subdir)
    keys_path = os.path.join(KEYS_DIR, KEYS_FILENAME)
    keys = transform_streaming(
        model_name, out_dir, seed,
        alpha_e=alpha_e, alpha_h=alpha_h, beta=beta, zeta=zeta,
        keys_path=keys_path, rope_scaling=rope_scaling,
        obfuscate_attention=obfuscate_attention,
        apply_permutation=apply_permutation,
    )
    models_vol.commit()
    keys_vol.commit()
    import hashlib
    with open(keys_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "model_dir": out_dir,
        "keys_path": keys_path,
        "keys_sha256": digest,
        "seed": seed,
        "alpha_e": alpha_e, "alpha_h": alpha_h, "beta": beta, "zeta": zeta,
        "model_name": model_name,
        "out_subdir": out_subdir,
        "obfuscate_attention": obfuscate_attention,
        "apply_permutation": apply_permutation,
    }


@app.function(
    image=TRANSFORM_IMAGE,
    memory=TRANSFORM_MEMORY,
    ephemeral_disk=TRANSFORM_EPHEMERAL_DISK,  # téléchargement du source HF (16 Go)
    volumes={MODELS_DIR: models_vol},
    timeout=3600,
    scaledown_window=300,
)
def verify(
    model_subdir: str = MODEL_SUBDIR,
    seed: int = 0,
    samples: int = 16,
    layers: int = 2,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    beta: int = 8,
    gamma: float = 1e3,
    zeta: float = 1e3,
    rope_scaling: str = "auto",
):
    """Vérification bit-à-bit par échantillons du modèle obfusqué.

    Charge `aloepri/verify_transform.py` et compare des échantillons de lignes
    embed/head et de couches complètes entre le modèle source (cache HF,
    `SRC_MODEL`) et le modèle obfusqué sous
    `{MODELS_DIR}/{model_subdir}` sur le Volume. Les clés sont RÉGÉNÉRÉES par
    seed (même tirage que `transform_streaming`) : aucun Volume clés monté,
    la clé de permutation reste côté client.

    Les hyperparamètres (alpha_e, alpha_h, beta, gamma, zeta, rope_scaling)
    doivent correspondre à ceux utilisés par `transform()`.
    """
    import json

    sys.path.insert(0, "/pkg/aloepri")
    from aloepri import verify_transform
    from aloepri.transform_streaming import _vocab_permutation

    model_dir = os.path.join(MODELS_DIR, model_subdir)
    src_dir = verify_transform._source_paths(SRC_MODEL)
    with open(os.path.join(src_dir, "config.json")) as f:
        src_cfg = json.load(f)
    with open(os.path.join(model_dir, "config.json")) as f:
        out_cfg = json.load(f)

    # clés régénérées par seed (identiques à celles de transform_streaming)
    permutation, unpermute, _ = _vocab_permutation(
        seed, src_cfg["vocab_size"])
    keys = {
        "vocab_permutation": {str(k): int(v) for k, v in permutation.items()},
        "vocab_unpermute": {str(k): int(v) for k, v in unpermute.items()},
        "seed": seed,
    }

    # bool strict — même conversion que transform_streaming : une chaîne
    # "off"/"on" ne doit pas être truthy (verify(rope_scaling="off") doit
    # désactiver Ĥ exactement comme transform(rope_scaling="off")).
    if rope_scaling == "on":
        rope_scaling = True
    elif rope_scaling == "off":
        rope_scaling = False
    else:
        rope_scaling = (src_cfg.get("model_type") not in
                        ("qwen3", "qwen3_moe"))

    problems = []
    problems += verify_transform.check_structure(
        model_dir, src_cfg["vocab_size"], src_cfg["num_hidden_layers"])
    problems += verify_transform.check_embedding_rows(
        model_dir, src_dir, keys, samples, alpha_e, alpha_h)
    problems += verify_transform.check_layers(
        model_dir, src_dir, keys, layers, beta, gamma, zeta, rope_scaling)

    # IDs spéciaux dans l'espace permuté (même contrôle que le module)
    for field in ("bos_token_id", "eos_token_id"):
        src_id = src_cfg.get(field)
        if src_id is None:
            continue
        exp = keys["vocab_permutation"][str(src_id)]
        got = out_cfg.get(field)
        if got != exp:
            problems.append(f"config.{field}: attendu {exp} (source {src_id}), "
                            f"obtenu {got}")

    print("== Vérification du modèle obfusqué ==")
    print(f"  modèle : {model_dir}  ("
          f"{len(verify_transform._weight_map(model_dir))} tenseurs)")
    print(f"  échantillons recomputés : {samples} lignes embed/head, "
          f"{layers} couches")
    if problems:
        print(f"  [KO] {len(problems)} problème(s) :")
        for p in problems[:20]:
            print(f"    - {p}")
    else:
        print("  [OK] structure, config, dtype, échantillons bit-à-bit "
              "conformes")
    return {
        "model_dir": model_dir,
        "ok": not problems,
        "n_problems": len(problems),
        "problems": problems[:20],
        "samples": samples,
        "layers": layers,
    }


@app.function(
    image=SERVE_IMAGE,
    gpu=GPU_SERVE,
    volumes={MODELS_DIR: models_vol},
    secrets=[API_SECRET] if API_SECRET else [],
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    """Serveur HTTP du modèle obfusqué — posture STRICTE.

    Un seul endpoint, `POST /generate` : il reçoit des IDs de tokens PERMUTÉS
    (des nombres) et renvoie des IDs PERMUTÉS. Aucun tokenizer, aucune clé,
    aucun texte sur le serveur — la permutation du vocabulaire reste
    exclusivement côté client.

    Les paramètres de décodage optionnels (`repetition_penalty`,
    `bad_words_ids`) sont opaques pour le serveur : ce sont des nombres
    fournis par le client pour piloter `generate()` sans rien révéler.

    `@modal.asgi_app()` : la fonction RETOURNE l'app FastAPI (le conteneur
    est prêt dès qu'elle est retournée) — avec `@modal.web_server` il
    faudrait lancer uvicorn en sous-processus et rendre la main (pattern
    tiron), sinon la passerelle renvoie 303 tant que la fonction n'est pas
    retournée."""
    import torch
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM

    model_dir = os.path.join(MODELS_DIR, MODEL_SUBDIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).cuda()
    model.eval()

    fastapi_app = FastAPI()

    class GenerateRequest(BaseModel):
        input_ids: list[int]
        # max_new_tokens borné (1..2048) : pas de décodage infini ni de
        # valeurs absurdes côté client.
        max_new_tokens: int = Field(default=100, ge=1, le=2048)
        repetition_penalty: float = 1.0
        bad_words_ids: list[list[int]] | None = None

    class GenerateResponse(BaseModel):
        output_ids: list[int]

    def _authorized(authorization: str | None) -> bool:
        # Fail-closed : sans clé attendue configurée (env `ALOEPRI_API_KEY`
        # du Secret Modal `aloepri-api-key`), AUCUNE requête n'est acceptée —
        # le Secret est requis, cf. README (posture de sécurité).
        expected = os.environ.get("ALOEPRI_API_KEY")
        if not expected:
            return False
        return authorization == f"Bearer {expected}"

    @fastapi_app.get("/health")
    def health(authorization: str | None = Header(default=None)):
        if not _authorized(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")
        return {"status": "ok"}

    @fastapi_app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest,
                 authorization: str | None = Header(default=None)):
        if not _authorized(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")
        input_tensor = torch.tensor([req.input_ids], device=model.device)
        kwargs = {}
        if req.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = req.repetition_penalty
        if req.bad_words_ids:
            kwargs["bad_words_ids"] = req.bad_words_ids
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=min(req.max_new_tokens, 2048),  # clamp défensif
                do_sample=False, **kwargs,
            )
        return GenerateResponse(output_ids=output[0].tolist())

    return fastapi_app


@app.function(
    image=TRANSFORM_IMAGE,  # embarque /pkg/aloepri (isa_attack)
    gpu="A100-40GB",        # 16 Go de modèle + table fp32 + rétroprop > L4
    volumes={MODELS_DIR: models_vol},
    timeout=3600,
    scaledown_window=300,
)
def isa_attack(
    ids: str,
    channel: str = "hidden",
    layer: int = 18,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    model_ref: str = MODEL_SUBDIR,
):
    """Démonstration ISA par gradient (cf. `aloepri/isa_attack.py`) sur le
    modèle obfusqué RÉEL servi sur le Volume.

    L'attaque n'utilise QUE les poids obfusqués (l'attaquant = opérateur du
    serveur, sans clé). `ids` est l'entrée réelle du modèle — les IDs PERMUTÉS
    du prompt secret, en chaîne CSV — calculée CÔTÉ CLIENT avec les clés
    (jamais envoyées). La fonction renvoie les IDs récupérés ; la mesure du
    taux et la dépermutation se font côté client.

    `model_ref` : sous-répertoire sur le Volume `obfuscator-models`
    (`{MODELS_DIR}/{model_ref}`) par défaut ; préfixe `"hf:"` pour charger un
    modèle clair de HuggingFace (ex. `"hf:Qwen/Qwen3-0.6B"`) — aucun Volume
    clés monté dans tous les cas.
    """
    import json

    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, "/pkg/aloepri")
    from aloepri.isa_attack import run_channel_attack

    ids = [int(x) for x in ids.split(",")]
    if model_ref.startswith("hf:"):
        model_dir = model_ref[len("hf:"):]
    else:
        model_dir = os.path.join(MODELS_DIR, model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()

    pred, rate, losses = run_channel_attack(
        model, ids, channel, layer, steps=steps, lr=lr, seed=seed,
        device="cuda",
    )
    result = {
        "channel": channel,
        "layer": layer,
        "steps": steps,
        "model_ref": model_ref,
        "ids_envoyes_au_modele": len(ids),
        "taux_recuperation_ids_modele": rate,
        "ids_recuperes": pred.tolist(),
        "loss_debut": losses[0],
        "loss_fin": losses[-1],
    }
    print("RESULTAT_ISA", json.dumps(result), flush=True)
    return result


@app.function(volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
              scaledown_window=60)
def diag():
    """État des Volumes : contenu du modèle de service et des clés."""
    for d in (MODELS_DIR, KEYS_DIR):
        print(f"== {d} ==")
        for root, _dirs, files in os.walk(d):
            for fname in files:
                path = os.path.join(root, fname)
                print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} Mo)")
    if os.path.exists(os.path.join(KEYS_DIR, KEYS_FILENAME)):
        import hashlib
        with open(os.path.join(KEYS_DIR, KEYS_FILENAME), "rb") as f:
            print("keys sha256:",
                  hashlib.sha256(f.read()).hexdigest())
