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


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol,
              "/poc": modal.Volume.from_name("aloepri-models", create_if_missing=False)},
              scaledown_window=60)
def compare_poc():
    """Diagnostic : compare les poids du modèle obfusqué local (obfuscator-models)
    avec le modèle POC (aloepri-models) — même structure attendue si les
    transformations sont identiques (seed 0, alpha_e, beta)."""
    import json
    import torch
    from safetensors import safe_open

    poc = "/poc/qwen3-8b-obf"
    obf = os.path.join(MODELS_DIR, MODEL_SUBDIR)
    idx = json.load(open(os.path.join(poc, "model.safetensors.index.json")))
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    ]
    for name in names:
        shard = idx["weight_map"].get(name)
        if not shard:
            print(f"{name}: introuvable dans l'index"); continue
        with safe_open(os.path.join(poc, shard), framework="pt") as fp, \
             safe_open(os.path.join(obf, shard), framework="pt") as fo:
            p = fp.get_tensor(name); o = fo.get_tensor(name)
        if p.shape != o.shape:
            print(f"{name}: SHAPE {p.shape} vs {o.shape}"); continue
        if torch.equal(p, o):
            print(f"{name}: IDENTIQUE")
        else:
            d = float((p.float() - o.float()).abs().max())
            print(f"{name}: DIFF max={d:.6g}")
    pc = json.load(open(os.path.join(poc, "config.json")))
    oc = json.load(open(os.path.join(obf, "config.json")))
    print("config identiques:", pc == oc)

@app.function(image=SERVE_IMAGE, scaledown_window=60)
def serve_env():
    """Diagnostic : versions des libs dans le conteneur serveur."""
    import torch
    import transformers
    import safetensors
    print(f"transformers={transformers.__version__}")
    print(f"torch={torch.__version__}")
    print(f"safetensors={safetensors.__version__}")


@app.function(image=TRANSFORM_IMAGE, gpu="A100-40GB",
              volumes={MODELS_DIR: models_vol},
              timeout=3600, scaledown_window=300)
def attention_inversion(
    ids: str,
    layer: int = 0,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
):
    """Attaque « clean-space » : capture les scores d'attention du modèle
    obfusqué (serveur) puis inverse dans le modèle baseline PUBLIC (poids
    gelés) → tokens récupérés EN CLAIR (contourne la permutation de vocabulaire).

    `ids` : les ids PERMUTÉS du prompt secret (l'entrée réelle du modèle),
    en CSV — calculés côté client avec les clés (jamais envoyées)."""
    import json
    import sys as _sys
    import torch
    from transformers import AutoModelForCausalLM

    _sys.path.insert(0, "/pkg/aloepri")
    from attention_inversion import run_attention_inversion

    ids = [int(x) for x in ids.split(",")]
    model_dir = os.path.join(MODELS_DIR, MODEL_SUBDIR)
    obf = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    clean = AutoModelForCausalLM.from_pretrained(
        SRC_MODEL, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    pred, losses = run_attention_inversion(
        obf, clean, ids, layer=layer, steps=steps, lr=lr, seed=seed,
        device="cuda",
    )
    result = {
        "layer": layer,
        "ids_envoyes_au_modele": len(ids),
        "ids_cles_recuperes": pred.tolist(),
        "loss_debut": losses[0],
        "loss_fin": losses[-1],
    }
    print("RESULTAT_INVERSION " + json.dumps(result), flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE,
              volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def vma_attack(subset_size: int = 2000, seed: int = 0,
               model_subdir: str = MODEL_SUBDIR):
    """VMA (Appendice D) sur le modèle 8B réel du Volume.

    L'attaquant (= opérateur du serveur) a les poids obfusqués du Volume et
    le modèle de base PUBLIC (Qwen3-8B). Il apparie les lignes de la table
    d'embedding obfusquée aux lignes de la table claire par plus proche
    voisin (cosinus) → récupère la permutation de vocabulaire Π, sans aucune
    clé. La vérité terrain (perm) est régénérée par seed UNIQUEMENT pour
    MESURER la récupération.

    Variante RowSort (mécanisme du papier, pour éliminer une permutation de
    colonnes Z2) mesurée aussi. Diagnostic : cosinus du vrai match vs meilleur
    faux match — interprète le taux de récupération."""
    import json
    import random
    import sys as _sys
    import urllib.request

    import torch
    from safetensors import safe_open

    _sys.path.insert(0, "/pkg/aloepri")
    from vma_attack import run_vma

    model_dir = os.path.join(MODELS_DIR, model_subdir)
    obf_embed = None
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fname), framework="pt") as fp:
            if "model.embed_tokens.weight" in fp.keys():
                obf_embed = fp.get_tensor(
                    "model.embed_tokens.weight").float().cpu()
                break
    assert obf_embed is not None, \
        "model.embed_tokens.weight introuvable sur le Volume"

    # table claire : SEUL le shard contenant embed_tokens (léger vs 16 Go)
    idx_url = f"https://huggingface.co/{SRC_MODEL}/resolve/main/" \
              "model.safetensors.index.json"
    import json as _json
    with urllib.request.urlopen(idx_url) as r:
        index = _json.loads(r.read().decode())
    shard = index["weight_map"]["model.embed_tokens.weight"]
    from huggingface_hub import hf_hub_download
    clear_path = hf_hub_download(SRC_MODEL, shard)
    with safe_open(clear_path, framework="pt") as fp:
        clear_embed = fp.get_tensor("model.embed_tokens.weight").float()
    if obf_embed.shape != clear_embed.shape:
        # h>0 : la table obfusquée vit en d+2h ≠ d — la VMA DIRECTE est
        # structurellement impossible (les dimensions ne correspondent pas).
        result = {
            "modele": model_subdir,
            "taille_table_obf": obf_embed.shape[0],
            "dim_obf": obf_embed.shape[1],
            "dim_claire": clear_embed.shape[1],
            "vma_directe_possible": False,
            "explication": "h>0 : embedding en d+2h — comparaison directe "
                           "impossible (défense structurelle)",
        }
        print("RESULTAT_VMA " + json.dumps(result), flush=True)
        return result

    # vérité terrain : permutation régénérée par seed (même tirage que la
    # transform — `random.Random(seed).shuffle`)
    V = obf_embed.shape[0]
    rng_py = random.Random(seed)
    permuted_ids = list(range(V))
    rng_py.shuffle(permuted_ids)
    perm = dict(zip(range(V), permuted_ids))

    rate_cos, n = run_vma(obf_embed, clear_embed, perm,
                          subset_size=subset_size, seed=seed)
    rate_sort, _ = run_vma(obf_embed, clear_embed, perm,
                           subset_size=subset_size, seed=seed,
                           use_row_sort=True)

    # diagnostic : cosinus du vrai match vs meilleur faux match (sur le
    # sous-ensemble, table claire complète)
    torch.manual_seed(seed)
    clear_tokens = torch.randperm(V)[:subset_size]
    obf_rows = obf_embed[torch.tensor([perm[int(t)] for t in
                                       clear_tokens.tolist()])].float()
    q = torch.nn.functional.normalize(obf_rows, dim=1)
    t = torch.nn.functional.normalize(clear_embed, dim=1)
    sim = q @ t.t()                                  # (N, V)
    true_cos = sim[torch.arange(subset_size), clear_tokens].mean().item()
    sim[torch.arange(subset_size), clear_tokens] = float("-inf")
    best_false_cos = sim.max(dim=1).values.mean().item()

    result = {
        "modele": model_subdir,
        "taille_table": V,
        "subset_size": n,
        "taux_recuperation_cosinus_direct": round(rate_cos, 4),
        "taux_recuperation_row_sort": round(rate_sort, 4),
        "cos_vrai_match_moyen": round(true_cos, 4),
        "cos_meilleur_faux_match_moyen": round(best_false_cos, 4),
    }
    print("RESULTAT_VMA " + json.dumps(result), flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, memory=49152,
              ephemeral_disk=TRANSFORM_EPHEMERAL_DISK,
              volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
              timeout=10800, scaledown_window=300)
def transform_chained(
    seed: int = 0,
    alpha_e: float = 0.3,
    alpha_h: float = 0.2,
    lam: float = 0.3,
    h: int = 128,
    beta: int = 8,
    gamma: float = 1e3,
    zeta: float = 1e3,
    kappa_mode: str = "empirical",
    model_name: str = SRC_MODEL,
    out_subdir: str = "qwen3-8b-obf-h128",
):
    """Schéma COMPLET AloePri (h>0, §5.4) : chaînage global P̂/Q̂, d → d+2h.

    Le modèle obfusqué est RECONSTRUIT avec `hidden_size = d + 2h` (le modèle
    clair n'est pas modifié) : chaque poids de la frontière hidden est
    transformé (embed·P̂, q/k/v/gate/up·Q̂ᵀ avec Wnorm fusionné, o/down·P̂,
    head·Q̂ᵀ), les RMSNorm deviennent des normes à poids scalaire κ (§5.2.5,
    κ empirique par couche par défaut — l'hypothèse gaussienne du papier
    biaise l'échelle, mesuré sur Qwen3-0.6B).

    Sortie : modèle obfusqué (hidden d+2h) sous
    `{MODELS_DIR}/{out_subdir}` + clés sur le Volume `obfuscator-keys`.
    Mémoire : modèle clair bf16 (~16 Go) + obfusqué bf16 (~17,5 Go) +
    transitoires fp64 → 48 Go demandés.
    """
    import json
    import sys as _sys

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from aloepri.chained_transform import obfuscate_chained
    from transformers import AutoModelForCausalLM

    clear = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager").eval()
    # le modèle obfusqué est construit directement en bf16 (le constructeur
    # créerait 34 Go de fp32 sinon) ; les calculs restent en fp64 explicites.
    torch.set_default_dtype(torch.bfloat16)
    try:
        obf, keys = obfuscate_chained(
            clear, clear.config, seed, alpha_e=alpha_e, alpha_h=alpha_h,
            lam=lam, h=h, beta=beta, gamma=gamma, zeta=zeta,
            kappa_mode=kappa_mode)
    finally:
        torch.set_default_dtype(torch.float32)
    del clear

    out_dir = os.path.join(MODELS_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    obf.to(torch.bfloat16).save_pretrained(out_dir)

    keys_path = os.path.join(KEYS_DIR, KEYS_FILENAME)
    with open(keys_path, "w") as f:
        json.dump({k: ({str(a): int(b) for a, b in v.items()}
                       if isinstance(v, dict) else v)
                   for k, v in keys.items()}, f)
    models_vol.commit()
    keys_vol.commit()

    return {
        "out_subdir": out_subdir,
        "hidden_size": obf.config.hidden_size,
        "seed": seed, "alpha_e": alpha_e, "alpha_h": alpha_h,
        "lam": lam, "h": h, "beta": beta, "kappa_mode": kappa_mode,
        "model_name": model_name,
    }


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def verify_chained(
    model_subdir: str = "qwen3-8b-obf-h128",
    seed: int = 0,
    h: int = 128,
    n_prompts: int = 8,
):
    """Contrôle qualité du modèle chaîné h>0 (le round-trip est APPROXIMATIF :
    erreur κ §5.2.5 — pas de vérification bit-à-bit).

    Charge le modèle obfusqué du Volume + le modèle clair HF, compare les
    logits (corrélation + taux top-1) sur des prompts de test, et génère
    3 questions canoniques (capitale / 391+2 / haïku) — le gate de qualité
    avant de servir le modèle h>0.
    """
    import json
    import sys as _sys

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = os.path.join(MODELS_DIR, model_subdir)
    obf = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    clear = AutoModelForCausalLM.from_pretrained(
        SRC_MODEL, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    tok = AutoTokenizer.from_pretrained(SRC_MODEL)

    import random as _random
    rng_py = _random.Random(seed)
    permuted = list(range(clear.config.vocab_size))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(clear.config.vocab_size), permuted))
    unperm = {v: k for k, v in perm.items()}

    prompts = [
        "La capitale de la France est",
        "Combien font 391 + 2 ?",
        "Écris un haïku sur la mer.",
        "Le chat dort sur le canapé.",
        "Quelle est la couleur du ciel ?",
        "Raconte une courte histoire.",
        "Qui a écrit Les Misérables ?",
        "Traduis bonjour en anglais.",
    ][:n_prompts]

    results = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            lc = clear(ids).logits.double()
            p_ids = torch.tensor([[perm[int(t)] for t in ids[0]]]).cuda()
            lo = obf(p_ids).logits.double()
            cols = torch.tensor(
                [perm[t] for t in range(clear.config.vocab_size)]).cuda()
            lo_p = lo[..., cols]
            corr = torch.stack([lo_p.flatten(), lc.flatten()]
                               ).corrcoef()[0, 1].item()
            top1 = float((lo_p[0, -1].argmax() == lc[0, -1].argmax()).item())
            results.append({"prompt": p, "corr": round(corr, 4),
                            "top1": top1})

        def gen(text):
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            p = torch.tensor([[perm[int(t)] for t in ids[0]]]).cuda()
            out = obf.generate(
                p, max_new_tokens=40, do_sample=False,
                repetition_penalty=1.05,
                bad_words_ids=[[perm[151667]]],
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )[0].tolist()
            return tok.decode([unperm.get(int(x), x) for x in out],
                              skip_special_tokens=True)

        gen_results = {
            "capitale": gen("La capitale de la France est"),
            "391+2": gen("Combien font 391 + 2 ?"),
            "haiku": gen("Écris un haïku sur la mer."),
        }

    result = {
        "model_subdir": model_subdir,
        "hidden_size": obf.config.hidden_size,
        "logits": results,
        "top1_moyen": round(sum(r["top1"] for r in results) / len(results), 4),
        "generation": gen_results,
    }
    print("RESULTAT_VERIFY_CHAINED " + json.dumps(result, ensure_ascii=False),
          flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def vma_product_attack(
    model_subdir: str = "qwen3-8b-obf-h128",
    seed: int = 0,
    subset_size: int = 2000,
    layers: str = "0,4,8,12,16,20,24,28",
):
    """VMA PRODUIT (Table 9, Appendice D) sur le modèle h>0.

    L'attaquant a les poids obfusqués (Volume) + le modèle public. La VMA
    directe est impossible (d+2h ≠ d) ; les PRODUITS W̃_e·W̃_gateᵀ annulent
    P̂/Q̂ (chaînage) et retombent dans un espace comparable ; RowSort élimine
    la permutation FFN Ẑ_ffn ; l'appariement des lignes triées récupère Π.
    Seul le bruit d'embedding α_e (corrélé entre couches) le défend — la
    Table 3 du papier mesure TTRSR 25 % (Qwen3-14B) / 20 % (32B) à α_e=1.0.
    """
    import json
    import random
    import sys as _sys
    import urllib.request

    import torch
    from safetensors import safe_open

    _sys.path.insert(0, "/pkg/aloepri")
    from vma_product import run_vma_product

    layer_list = [int(x) for x in layers.split(",")]
    model_dir = os.path.join(MODELS_DIR, model_subdir)

    def _load_from_dir(directory, names):
        out = {}
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(directory, fname),
                           framework="pt") as fp:
                for n in names:
                    if n in fp.keys() and n not in out:
                        out[n] = fp.get_tensor(n)
        return out

    names = (["model.embed_tokens.weight"] +
             [f"model.layers.{i}.mlp.gate_proj.weight"
              for i in layer_list])
    obf_t = _load_from_dir(model_dir, names)
    obf_embed = obf_t["model.embed_tokens.weight"].float()
    obf_gates = [obf_t[f"model.layers.{i}.mlp.gate_proj.weight"].float()
                 for i in layer_list]

    import json as _json
    with urllib.request.urlopen(
            f"https://huggingface.co/{SRC_MODEL}/resolve/main/"
            "model.safetensors.index.json") as r:
        wm = _json.loads(r.read().decode())["weight_map"]
    from huggingface_hub import hf_hub_download

    clear_gates, wns = [], []
    for i in layer_list:
        gname = f"model.layers.{i}.mlp.gate_proj.weight"
        wname = f"model.layers.{i}.post_attention_layernorm.weight"
        for n in (gname, wname):
            path = hf_hub_download(SRC_MODEL, wm[n])
            with safe_open(path, framework="pt") as fp:
                if n == gname:
                    clear_gates.append(fp.get_tensor(n).float())
                else:
                    wns.append(fp.get_tensor(n).float())
    path = hf_hub_download(SRC_MODEL, wm["model.embed_tokens.weight"])
    with safe_open(path, framework="pt") as fp:
        clear_embed = fp.get_tensor("model.embed_tokens.weight").float()

    V = clear_embed.shape[0]
    rng_py = random.Random(seed)
    permuted_ids = list(range(V))
    rng_py.shuffle(permuted_ids)
    perm = dict(zip(range(V), permuted_ids))

    # les tenseurs chargés par safe_open sont sur CPU : tout passe sur le GPU
    # (les produits V×inter et les RowSort sont des centaines de fois plus
    # rapides sur A100 — le run précédent ramait sur CPU faute de .cuda()).
    # Grandes tables en bf16 (X clair V×inter = 3,7 Go au lieu de 7,5 Go en
    # fp32) : le run fp32 a fait un OOM CUDA sur les 40 Go de l'A100.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obf_embed = obf_embed.to(dev)
    obf_gates = [g.to(dev) for g in obf_gates]
    clear_embed = clear_embed.to(dev)
    clear_gates = [g.to(dev) for g in clear_gates]
    wns = [w.to(dev) for w in wns]

    res = run_vma_product(
        obf_embed, obf_gates, clear_embed, clear_gates, wns, perm,
        subset_size=subset_size, seed=seed, dtype=torch.bfloat16)
    result = {"modele": model_subdir, "layers": layer_list, **res}
    print("RESULTAT_VMA_PRODUIT "
          + json.dumps(result, ensure_ascii=False), flush=True)
    return result
