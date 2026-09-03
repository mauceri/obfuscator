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
        "safetensors", "huggingface_hub", "datasets",
    )
    .add_local_dir(_ALOEPRI_DIR, "/pkg/aloepri", copy=True)
    .add_local_file(
        "/home/mauceric/gen_corpus_gepa_codex/corpus_synth_clean_10000.jsonl",
        "/corpus_gepa.jsonl", copy=True)
    # Échantillon frwiki (NDJSON, 10 fichiers wiki_XX) pour la précision 9.6 :
    # le corpus complet (~/corpus_fr/frwiki, 1,1 Go) n'est pas monté dans le
    # conteneur — on embarque un sous-ensemble déterministe (wiki_00..09).
    # copy=True partout : tous les add_local_* copient dans l'image au build
    # (un mount non-copied suivi d'un build step est refusé par Modal).
    .add_local_dir(
        "/home/mauceric/corpus_fr/frwiki_sample",
        "/frwiki_sample", copy=True)
    # Vocabulaire effectif du corpus GEPA (ids triés par fréquence décroissante,
    # un par ligne) — pour l'analyse « où tombent les tokens récupérés » (Zipf).
    .add_local_file(
        "/home/mauceric/obfuscator/data/gepa_vocab_ids.txt",
        "/gepa_vocab_ids.txt", copy=True)
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


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def isa_vocab_attack(
    ids: str,
    layer: int = 18,
    k: int = 64,
    channel: str = "hidden",
    teacher_forcing: bool = True,
    metric: str = "mse",
    seed: int = 0,
    model_ref: str = MODEL_SUBDIR,
):
    """ISA DISCRÈTE (vocab-matching k-way) — variante du point 4 de la revue.

    La variante par gradient (isa_attack) trouve des soft tokens dans
    l'enveloppe convexe du simplexe qui reproduisent l'état caché sans être
    le prompt (loss → 0,007 mais ids ≠ prompt) : elle ne permet pas de
    conclure « canal hidden non informatif ». Ici, recherche DISCRÈTE : à
    chaque position, le vrai token est mélangé à k−1 leurres et le canal
    doit l'identifier (taux → 100 % si informatif, → 1/k si sous-déterminé).

    `ids` : IDs PERMUTÉS du prompt secret (côté client, clé seed 0).
    `teacher_forcing` : True = préfixe réel (borne haute, isole le canal) ;
    False = greedy autorégressif (récupération réelle).
    `metric` : "mse" (relatif) ou "cos" (insensible à l'échelle).
    """
    import json

    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, "/pkg/aloepri")
    from aloepri.isa_attack import run_vocab_match

    ids = [int(x) for x in ids.split(",")]
    if model_ref.startswith("hf:"):
        model_dir = model_ref[len("hf:"):]
    else:
        model_dir = os.path.join(MODELS_DIR, model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()

    pred, rate, hit = run_vocab_match(
        model, ids, layer=layer, k=k, channel=channel,
        teacher_forcing=teacher_forcing, metric=metric, seed=seed,
        device="cuda",
    )
    result = {
        "channel": channel,
        "layer": layer,
        "k": k,
        "teacher_forcing": teacher_forcing,
        "metric": metric,
        "model_ref": model_ref,
        "ids_envoyes_au_modele": len(ids),
        "taux_identification_kway": rate,
        "ids_recuperes": pred.tolist(),
    }
    print("RESULTAT_ISA_VOCAB " + json.dumps(result), flush=True)

    # résumé lisible
    baseline = 1.0 / k
    print("=" * 62, flush=True)
    print("RÉSUMÉ — ISA discrète (vocab-matching k-way)", flush=True)
    print(f"  modèle   : {model_ref}", flush=True)
    print(f"  canal    : {channel}, couche {layer}", flush=True)
    print(f"  k        : {k} candidats/position "
          f"(baseline aléatoire = {baseline:.1%})", flush=True)
    mode_txt = ("teacher-forced (borne haute)" if teacher_forcing
               else "greedy autorégressif")
    print(f"  mode     : {mode_txt}", flush=True)
    print(f"  métrique : {metric}", flush=True)
    print(f"  taux     : {rate:.1%}", flush=True)
    interp = ("canal INFORMATIF (le vrai token est distingué)"
              if rate > 2 * baseline else "canal sous-déterminé")
    print(f"  ➜ {interp}", flush=True)
    print("=" * 62, flush=True)
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


@app.function(image=TRANSFORM_IMAGE, memory=131072,
              ephemeral_disk=TRANSFORM_EPHEMERAL_DISK,
              volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
              timeout=14400, scaledown_window=300)
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
    Mémoire : modèle clair bf16 (~16 Go pour le 8B, ~28 Go pour le 14B) +
    obfusqué bf16 (~17,5/30 Go) + transitoires fp64 → 128 Go demandés
    (couvre le 8B ET le 14B ; re-abaisser à 49152 pour ne transformer que
    le 8B et réduire le coût CPU).
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
              gpu="A100-80GB", timeout=3600, scaledown_window=300)
def verify_chained(
    model_subdir: str = "qwen3-8b-obf-h128",
    seed: int = 0,
    h: int = 128,
    n_prompts: int = 8,
    base_ref: str = SRC_MODEL,
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
        base_ref, dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    tok = AutoTokenizer.from_pretrained(base_ref)

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


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-80GB", timeout=7200, scaledown_window=300)
def finetune_corpus(
    model_name: str = "Qwen/Qwen3-0.6B",
    epochs: int = 5,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 2e-5,
    out_subdir: str = "qwen3-06b-ft-gepa",
    seed: int = 0,
):
    """Entraînement COMPLET (fine-tuning de tous les paramètres) sur le corpus
    GEPA (français synthétique hors distribution), GPU A100-80GB.

    Mémoire (Qwen3-8B, 8,07 Md params, bf16 complet) : poids 16,1 Go +
    gradients 16,1 Go + états AdamW (m/v) 32,3 Go = 64,5 Go, + activations
    avec gradient checkpointing (~2-4 Go) → tient sur 80 Go. AdamW fp32
    classique (m/v = 64,6 Go) porterait le total à 96,8 Go — infaisable,
    même sur A100-80GB.

    Objectif (spike VMA) : un vrai entraînement doit décaler les poids assez
    pour casser la VMA produit contre la référence publique. Sortie : modèle
    fine-tuné sur le Volume `obfuscator-models/{out_subdir}` + courbe de loss.
    """
    import json
    import sys as _sys
    import time

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # corpus
    texts = []
    with open("/corpus_gepa.jsonl") as f:
        for line in f:
            d = json.loads(line)
            c = d.get("contenu")
            if c and isinstance(c, str) and len(c) > 50:
                texts.append(c)
    print(f"[corpus] {len(texts)} textes", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    all_ids = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            all_ids.extend(ids + [tok.eos_token_id])
    n_seq = len(all_ids) // seq_len
    corpus = torch.tensor(all_ids[:n_seq * seq_len]).view(n_seq, seq_len)
    print(f"[corpus] {len(all_ids)} tokens → {n_seq} séquences de {seq_len}",
          flush=True)

    torch.manual_seed(seed)
    # bf16 COMPLET (poids + grads + états AdamW) : avec des params bf16,
    # torch.optim.AdamW crée des états m/v bf16 automatiquement (zeros_like).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16).to("cuda").train()
    model.config.use_cache = False  # requis avec gradient checkpointing
    model.gradient_checkpointing_enable()  # activations re-calculées au backward
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    steps_per_epoch = max(1, n_seq // batch_size)
    total = epochs * steps_per_epoch
    t0 = time.time()
    losses = []
    for step in range(total):
        idx = torch.randint(0, n_seq, (batch_size,))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(corpus[idx].cuda(), labels=corpus[idx].cuda())
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == total - 1:
            print(f"[ft] step {step}/{total} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    out_dir = os.path.join(MODELS_DIR, out_subdir)
    model.eval().to("cpu")
    model.save_pretrained(out_dir)
    models_vol.commit()
    print(f"[ft] loss début={losses[0]:.4f} fin={losses[-1]:.4f} "
          f"min={min(losses):.4f} — {total} pas en "
          f"{(time.time()-t0)/60:.1f} min → {out_dir}", flush=True)
    return {
        "out_subdir": out_subdir,
        "steps": total,
        "loss_debut": losses[0],
        "loss_fin": losses[-1],
        "loss_min": min(losses),
        "minutes": round((time.time() - t0) / 60, 1),
        "model_name": model_name,
    }


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              timeout=1200, scaledown_window=60)
def compare_embed_sources(
    ft_ref: str = "qwen3-8b-ft-h128",
    base_ref: str = "qwen3-8b-base-h128-a0",
    n: int = 20000,
    seed: int = 0,
):
    """Compare les embeddings de deux modèles obfusqués du volume (ligne à
    ligne : même permutation seed 0, mêmes P̂/Q̂).

    Objectif (diagnostic 2026-09-01) : vérifier que `qwen3-8b-ft-h128` a bien
    été construit depuis le FINE-TUNÉ et non depuis la base. Si cos ≈ 1 entre
    ft_h128 et base_h128_a0 (modulo le bruit α_e), le modèle servi est (quasi)
    la base obfusquée → l'attaque VMA contre la référence publique le récupère
    presque entièrement (90,8 %), sans que le fine-tuning n'ait rien apporté.
    """
    import random

    import torch
    from safetensors import safe_open

    def _embed(subdir):
        path = os.path.join(MODELS_DIR, subdir, "model.safetensors")
        with safe_open(path, framework="pt") as fp:
            return fp.get_tensor("model.embed_tokens.weight")

    a = _embed(ft_ref)
    b = _embed(base_ref)
    torch.manual_seed(seed)
    idx = torch.randperm(a.shape[0])[:n]
    aa = a[idx].float()
    bb = b[idx].float()
    cos = (aa * bb).sum(1) / (aa.norm(dim=1) * bb.norm(dim=1) + 1e-8)
    norm_a = aa.norm(dim=1).mean().item()
    norm_b = bb.norm(dim=1).mean().item()
    # mêmes lignes identiques ? (bruit α_e=0 vs 0,3 : les lignes de la base
    # sans bruit devraient être EXACTES si le FT n'a pas été appliqué)
    cos_moyen = float(cos.mean().item())
    print(f"[compare] {ft_ref} vs {base_ref} — cos moyen = {cos_moyen:.4f} "
          f"({n} lignes)", flush=True)
    print(f"[compare] norme moyenne {ft_ref} = {norm_a:.3f} | "
          f"{base_ref} = {norm_b:.3f}", flush=True)
    result = {"ft_ref": ft_ref, "base_ref": base_ref, "n": n,
              "cos_moyen": cos_moyen,
              "norme_ft": norm_a, "norme_base": norm_b}
    print("RESULTAT_COMPARE_EMBEDS " + __import__("json").dumps(result),
          flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=7200, scaledown_window=300)
def vma_product_full(
    model_subdir: str = "qwen3-8b-obf-h128",
    seed: int = 0,
    subset_size: int = 2000,
    views: str = "gate",
    sample: str = "uniform",
    clear_ref: str = SRC_MODEL,
):
    """VMA produit GRANDEUR NATURE (Table 9) : toutes les couches + agrégation.

    Le maximum réalisable à V=151 936 sur A100-40GB : la vue V×inter gate.
    COUVERTURE DES VUES DE LA TABLE 9 (état 2026-09-02) :

    | Vue | Structure Y = | Statut | Pourquoi |
    |---|---|---|---|
    | W_e·W_gate | Π·W*_e·W_gate·Ẑ_ffn | ✅ testée (cette fonction) | Ẑ (permutation) éliminé par RowSort — contrôle 99,95 % |
    | W_e·W_h | Π·W*_e·W_h·Πᵀ | ✅ testée (`vma_weh_attack`) | embed×head, V×V en streaming — contrôle 100 %, défensif 0 % |
    | W_e·W_up | Π·W*_e·W_up·Ŝ_ffn·Ẑ_ffn | ❌ écartée | Ŝ_ffn (scaling diagonal sur up_proj) N'EST PAS éliminé par RowSort → TTRSR ≈ 0 % acquis, sans rapport avec la défense |
    | W_down·W_h | Ẑ⁻¹·Ŝ⁻¹·W_down·W*_h·Πᵀ | ❌ écartée | porte Π sur les COLONNES : RowSort détruirait l'information à récupérer |
    | gram q·k | Π·W*_e·W_q·(Π·W*_e·W_k)ᵀ | ⏸️ en réserve | V×V (46 Go), deux produits croisés — non testée |
    | W_e·W_router | Π·W*_e·norm(W_router)·Ẑ_router | ➖ sans objet | MoE-only — Qwen3-8B dense |

    La vue up est donc INUTILISABLE par construction : le scaling Ŝ_ffn vit
    sur `up_proj` (ffn_obfuscation.py) et RowSort élimine Ẑ (permutation)
    mais PAS Ŝ (scaling diagonal, modifie le multiset de chaque ligne) →
    TTRSR ≈ 0 % acquis, sans rapport avec la défense.

    VOTE COMBINÉ : les vues testées peuvent être additionnées
    (`vma_combined_attack`) — chaque vue normalisée par z-score par ligne
    sur les V candidats (monotone → taux par vue inchangés), puis somme
    50/50 avant l'argmax. MESURÉ (2026-09-02) sur le défensif
    α_e=1,0/α_h=0,2 : gate 8,35 % / We·Wh 0 % / **combiné 6,55 %** — la
    vue gate seule reste la plus forte (We·Wh = bruit pur à ce réglage,
    il dilue le signal au lieu de l'ajouter). Contrôle : 100 %.

    Précision : les poids du volume sont en bf16, mais les produits
    W̃_e·W̃_gateᵀ sont calculés en FP32 — le chaînage P̂Q̂≈I en bf16 accumule
    une erreur (max|Y−X| ≈ 0,33) qui détruit l'appariement même à α_e=0
    (mesuré : 16 % au lieu de 100 % sur clés aléatoires).

    Pour chaque vue × chaque couche : produit W̃_e·W̃_gateᵀ (ou up), RowSort
    chunké, appariement cosinus sur `subset_size` tokens → similarités (N,V).
    Agrégation : somme des similarités (z-score par ligne) sur les couches,
    puis argmax global — le vote par mode (mode sur 36 quasi-aléatoires ≈ 0 %)
    détruisait le signal faible au lieu de l'agréger.
    """
    import json
    import random
    import sys as _sys
    import urllib.request

    import torch
    from safetensors import safe_open

    _sys.path.insert(0, "/pkg/aloepri")
    from vma_product import _row_sort_chunked
    from vma_attack import nearest_neighbor_rows

    view_list = [v.strip() for v in views.split(",")]
    model_dir = os.path.join(MODELS_DIR, model_subdir)

    def _tensors_from_dir(directory, names):
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

    # embed obfusqué + config
    obf_t = _tensors_from_dir(model_dir, ["model.embed_tokens.weight"])
    obf_embed = obf_t["model.embed_tokens.weight"]
    with open(os.path.join(model_dir, "config.json")) as f:
        n_layers = json.load(f)["num_hidden_layers"]

    # embed clair + poids clairs (HF, sélectif)
    import json as _json
    with urllib.request.urlopen(
            f"https://huggingface.co/{clear_ref}/resolve/main/"
            "model.safetensors.index.json") as r:
        wm = _json.loads(r.read().decode())["weight_map"]
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(clear_ref, wm["model.embed_tokens.weight"])
    with safe_open(path, framework="pt") as fp:
        clear_embed = fp.get_tensor("model.embed_tokens.weight")

    V = clear_embed.shape[0]
    rng_py = random.Random(seed)
    permuted_ids = list(range(V))
    rng_py.shuffle(permuted_ids)
    perm = dict(zip(range(V), permuted_ids))
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # échantillonnage des tokens testés :
    # - "uniform" : tirage uniforme sur le vocabulaire (défaut) ;
    # - "gepa-tete" : les `subset_size` tokens les plus fréquents de GEPA
    #   (fichier /gepa_vocab_ids.txt embarqué, ids triés par fréquence) —
    #   pour l'analyse « où tombent les récupérés » dans les classes
    #   fréquentes (1-10, 11-100, …) ;
    # - "gepa-strat" : tête (10 + 90) + échantillon uniforme du reste de GEPA.
    if sample == "uniform":
        test = torch.randperm(V)[:subset_size].to(dev)
    else:
        with open("/gepa_vocab_ids.txt") as f:
            gepa_ids = [int(l) for l in f if l.strip()]
        if sample == "gepa-tete":
            chosen = gepa_ids[:subset_size]
        elif sample == "gepa-strat":
            head = gepa_ids[:100]                       # classes 1-10 + 11-100
            rest = gepa_ids[100:]
            n_rest = subset_size - len(head)
            idx_rest = torch.randperm(len(rest),
                                      generator=torch.Generator()
                                      .manual_seed(seed))[:max(0, n_rest)]
            chosen = head + [rest[int(k)] for k in idx_rest.tolist()]
        else:
            raise ValueError(f"sample inconnu : {sample!r}")
        test = torch.tensor(chosen[:subset_size], device=dev)
    obf_rows_idx = torch.tensor([perm[int(t)] for t in test.tolist()]).to(dev)
    obf_embed = obf_embed.to(dev)
    clear_embed = clear_embed.to(dev)

    # poids par couche (chargés à la volée, libérés après)
    # agrégation : similarités (N, V) cumulées par vue (z-score par ligne
    # pour donner le même poids à chaque couche) puis argmax global
    per_view = {v: torch.zeros(len(test), V, dtype=torch.float32,
                               device=dev) for v in view_list}
    for i in range(n_layers):
        obf_names = [f"model.layers.{i}.mlp.gate_proj.weight",
                     f"model.layers.{i}.mlp.up_proj.weight"]
        obf_w = _tensors_from_dir(model_dir, obf_names)
        clr = {}
        for n in (f"model.layers.{i}.mlp.gate_proj.weight",
                  f"model.layers.{i}.mlp.up_proj.weight",
                  f"model.layers.{i}.post_attention_layernorm.weight"):
            p = hf_hub_download(clear_ref, wm[n])
            with safe_open(p, framework="pt") as fp:
                clr[n] = fp.get_tensor(n).to(dev)
        wn = clr[f"model.layers.{i}.post_attention_layernorm.weight"]

        for view in view_list:
            key = f"model.layers.{i}.mlp.{view}_proj.weight"
            g_obf = obf_w[key].to(dev).float()  # (inter, d2) obfusqué
            g_clair = (clr[key] * wn[None, :]).float()  # (inter, d) fold Wnorm
            # PRODUITS EN FP32 : le chaînage P̂Q̂≈I en bf16 accumule une erreur
            # (max|Y−X| ≈ 0,33) qui détruit l'appariement même à α_e=0.
            Y = (obf_embed[obf_rows_idx].float() @ g_obf.t())  # (N, inter)
            Y = _row_sort_chunked(Y)
            X = torch.empty(V, g_clair.shape[0],
                            dtype=torch.float32, device=dev)
            for c0 in range(0, V, 8192):
                c1 = min(c0 + 8192, V)
                X[c0:c1] = clear_embed[c0:c1].float() @ g_clair.t()
            X = _row_sort_chunked(X)
            # cosinus par blocs → similarités (N, V) ajoutées au cumul
            q = torch.nn.functional.normalize(Y, dim=1)
            t = torch.nn.functional.normalize(X, dim=1)
            for c0 in range(0, V, 8192):
                c1 = min(c0 + 8192, V)
                sim = q @ t[c0:c1].t()               # (N, 8192)
                sim = (sim - sim.mean(dim=1, keepdim=True)) \
                    / (sim.std(dim=1, keepdim=True) + 1e-6)   # z-score
                per_view[view][:, c0:c1] += sim
            rate = float((per_view[view].argmax(dim=1) == test)
                         .float().mean().item())
            print(f"[vma_full] vue={view} couche {i}/{n_layers}: "
                  f"TTRSR(cumul)={rate:.1%}", flush=True)
            del X, Y, q, t, sim
            if dev == "cuda":
                torch.cuda.empty_cache()
        del obf_w, clr

    # agrégation finale : argmax global sur le cumul de similarités
    result = {"modele": model_subdir, "vues": view_list,
              "n_couches": n_layers, "n_tokens": len(test)}
    all_preds = []
    for view, sims in per_view.items():
        pred = sims.argmax(dim=1)
        rate = float((pred == test).float().mean().item())
        result[f"vote_{view}"] = round(rate, 4)
        all_preds.append(sims)
    result["vote_global"] = round(float(
        (torch.stack(all_preds).sum(0).argmax(dim=1) == test)
        .float().mean().item()), 4)

    # tokens récupérés : les ids CLAIRS testés dont Π a été correctement
    # retrouvée (pred == test) — pour l'analyse de fréquence (Zipf) côté
    # client : où tombent-ils dans la distribution du corpus ?
    pred_global = torch.stack(all_preds).sum(0).argmax(dim=1)
    recup = test[pred_global == test].tolist()
    result["tokens_recuperes"] = recup
    result["n_tokens_recuperes"] = len(recup)
    print("RESULTAT_VMA_FULL " + json.dumps(result), flush=True)

    # résumé lisible (le JSON ci-dessus reste la sortie machine ; ce bloc est
    # destiné à l'œil humain — aucune valeur mesurée n'y est en dur)
    g = result["vote_global"] * 100
    interp = ("défense efficace : la permutation Π n'est PAS récupérée"
              if g < 1.0 else "défense défaillante : Π récupérée")
    print("=" * 62, flush=True)
    print("RÉSUMÉ — VMA produit (Table 9)", flush=True)
    print(f"  modèle   : {model_subdir}", flush=True)
    print(f"  vues     : {', '.join(view_list)} ({len(view_list)})", flush=True)
    print(f"  couches  : {n_layers} (agrégation : somme des similarités)", flush=True)
    print(f"  tokens   : {len(test)}", flush=True)
    for view in view_list:
        print(f"  taux {view:<12}: {result[f'vote_{view}'] * 100:.2f} %", flush=True)
    print(f"  taux global     : {g:.2f} %", flush=True)
    print(f"  ➜ {interp}", flush=True)
    print("=" * 62, flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=10800, scaledown_window=300)
def vma_weh_attack(
    model_subdir: str = "qwen3-8b-ft-h128-a1-h02",
    seed: int = 0,
    subset_size: int = 2000,
    sample: str = "uniform",
    block_rows: int = 2048,
    max_blocks: int = 0,
    clear_ref: str = SRC_MODEL,
):
    """VMA vue W_e·W_h (Table 9) — la dernière vue V×V, enfin testée.

    Ligne « WeWh » du papier : Y = W̃_e·W̃_headᵀ = Π·X·Πᵀ (+ bruits α_e/α_h)
    avec X = W_e·(wnorm_finale·W_h)ᵀ (référence publique). Le chaînage
    P̂Q̂=I élimine les clés de part et d'autre du produit ; RowSort par ligne
    élimine Π (à droite) ; on retrouve Π par appariement cosinus des lignes
    triées de Y (tokens testés) contre celles de X (vocabulaire clair).

    La table X est V×V (V=151 936 → 46 Go bf16) : jamais matérialisée —
    Y n'est calculé que pour `subset_size` lignes testées, et X est produit,
    trié puis apparié PAR BLOCS de `block_rows` lignes claires. Produits en
    FP32 (comme vma_product_full : le chaînage P̂Q̂≈I en bf16 accumule une
    erreur qui détruit l'appariement même à α=0).

    Le bruit se propage des DEUX côtés du produit (embed α_e ET head α_h) —
    contrairement à la vue gate (embed seul) — d'où l'intérêt de la mesure :
    α_h=0,2 (réglage papier Table 10) suffit-il, ou la vue fuit-elle plus
    que les 8,35 % de la gate à α_e=1,0 ?

    `max_blocks > 0` : traite seulement les N premiers blocs de X (calibrage
    du coût du RowSort V×V avant le run complet).

    Sortie : TTRSR (taux de récupération de Π) + tokens récupérés (pour
    l'analyse de fréquence Zipf) + timing par étape.

    Vote combiné : cette vue peut être additionnée à la gate dans
    `vma_combined_attack` (z-score par ligne puis somme 50/50). MESURÉ
    (2026-09-02) : contrôle 100 % ; au défensif α_e=1,0/α_h=0,2 la vue est
    du bruit pur (0 %) et le combiné (6,55 %) reste SOUS la gate seule
    (8,35 %) — voir la carte des vues dans la docstring de vma_product_full.
    """
    import json
    import random
    import sys as _sys
    import time
    import urllib.request

    import torch
    from safetensors import safe_open

    _sys.path.insert(0, "/pkg/aloepri")
    from vma_product import _row_sort_chunked

    model_dir = os.path.join(MODELS_DIR, model_subdir)

    def _tensors_from_dir(directory, names):
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

    # --- poids obfusqués (volume) : embed + head ---
    t0 = time.time()
    obf_t = _tensors_from_dir(model_dir, ["model.embed_tokens.weight",
                                          "lm_head.weight"])
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    V = cfg["vocab_size"]
    obf_embed = obf_t["model.embed_tokens.weight"]     # (V, d2) bf16
    obf_head = obf_t["lm_head.weight"]                  # (V, d2) bf16
    print(f"[weh] chargé obfusqué ({model_subdir}) : embed {tuple(obf_embed.shape)}"
          f", head {tuple(obf_head.shape)} — {time.time()-t0:.0f}s", flush=True)

    # --- poids clairs (référence publique HF) : embed + head + norme finale ---
    t0 = time.time()
    import json as _json
    with urllib.request.urlopen(
            f"https://huggingface.co/{clear_ref}/resolve/main/"
            "model.safetensors.index.json") as r:
        wm = _json.loads(r.read().decode())["weight_map"]
    from huggingface_hub import hf_hub_download

    def _hf_tensor(name):
        p = hf_hub_download(clear_ref, wm[name])
        with safe_open(p, framework="pt") as fp:
            return fp.get_tensor(name)

    clear_embed = _hf_tensor("model.embed_tokens.weight")     # (V, d) bf16
    clear_head = _hf_tensor("lm_head.weight")                  # (V, d) bf16
    wnorm = _hf_tensor("model.norm.weight")                    # (d,)
    print(f"[weh] chargé clair (HF {clear_ref}) — {time.time()-t0:.0f}s",
          flush=True)

    # --- permutation (seed) : convention random.Random(seed).shuffle ---
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))                       # {clair: obf}
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # échantillonnage des tokens testés (même logique que vma_product_full)
    if sample == "uniform":
        test = torch.randperm(V)[:subset_size].to(dev)
    else:
        with open("/gepa_vocab_ids.txt") as f:
            gepa_ids = [int(l) for l in f if l.strip()]
        if sample == "gepa-tete":
            chosen = gepa_ids[:subset_size]
        elif sample == "gepa-strat":
            head = gepa_ids[:100]
            rest = gepa_ids[100:]
            n_rest = subset_size - len(head)
            idx_rest = torch.randperm(len(rest),
                                      generator=torch.Generator()
                                      .manual_seed(seed))[:max(0, n_rest)]
            chosen = head + [rest[int(k)] for k in idx_rest.tolist()]
        else:
            raise ValueError(f"sample inconnu : {sample!r}")
        test = torch.tensor(chosen[:subset_size], device=dev)
    obf_rows = torch.tensor([perm[int(t)] for t in test.tolist()]).to(dev)
    n_tok = len(test)
    print(f"[weh] {n_tok} tokens testés (sample={sample})", flush=True)

    # --- Y = W̃_e[test]·W̃_headᵀ : (N, V) FP32, puis RowSort + normalize ---
    t0 = time.time()
    obf_embed = obf_embed.to(dev)
    obf_head = obf_head.to(dev)
    Y = obf_embed[obf_rows].float() @ obf_head.float().t()    # (N, V)
    del obf_embed, obf_head
    Y = _row_sort_chunked(Y)
    Y = torch.nn.functional.normalize(Y, dim=1)
    torch.cuda.empty_cache()
    print(f"[weh] Y ({tuple(Y.shape)}) produit+RowSort — {time.time()-t0:.0f}s",
          flush=True)

    # --- X par blocs de lignes claires : X_b = W_e[b]·(wnorm·W_h)ᵀ ---
    # A = (wnorm ∘ W_h)ᵀ : (d, V) fp32, précalculé une fois
    t0 = time.time()
    clear_embed = clear_embed.to(dev)
    A = (clear_head.float() * wnorm.float()[None, :]).t().to(dev)   # (d, V)
    del clear_head
    print(f"[weh] A=(wnorm·W_h)ᵀ {tuple(A.shape)} — {time.time()-t0:.0f}s",
          flush=True)

    best_sim = torch.full((n_tok,), float("-inf"), device=dev)
    best_idx = torch.zeros(n_tok, dtype=torch.long, device=dev)
    n_blocks = (V + block_rows - 1) // block_rows
    t_all = time.time()
    for b in range(n_blocks):
        b0, b1 = b * block_rows, min((b + 1) * block_rows, V)
        tb = time.time()
        Xb = clear_embed[b0:b1].float() @ A                      # (B, V)
        Xb = _row_sort_chunked(Xb)                               # RowSort
        Xn = torch.nn.functional.normalize(Xb, dim=1)
        sim = Y @ Xn.t()                                         # (N, B)
        sim_max, sim_arg = sim.max(dim=1)
        upd = sim_max > best_sim
        if upd.any():
            best_sim[upd] = sim_max[upd]
            best_idx[upd] = (b0 + sim_arg)[upd]
        del Xb, Xn, sim
        torch.cuda.empty_cache()
        if b % 5 == 0 or b == n_blocks - 1:
            print(f"[weh] bloc {b+1}/{n_blocks} ({b1} lignes) — "
                  f"{time.time()-tb:.1f}s (cumul X "
                  f"{time.time()-t_all:.0f}s)", flush=True)
        if max_blocks and b + 1 >= max_blocks:
            print(f"[weh] max_blocks={max_blocks} atteint — "
                  f"run de CALIBRAGE (temps/bloc → extrapolation)", flush=True)
            break

    rate = float((best_idx == test).float().mean().item())
    print(f"[weh] X complet traité — {time.time()-t_all:.0f}s au total",
          flush=True)

    result = {"modele": model_subdir, "vue": "weh", "n_tokens": n_tok,
              "block_rows": block_rows,
              "vote_weh": round(rate, 4),
              "tokens_recuperes": test[best_idx == test].tolist(),
              "n_tokens_recuperes": int((best_idx == test).sum().item())}
    print("RESULTAT_VMA_WEH " + json.dumps(result), flush=True)

    g = rate * 100
    interp = ("défense efficace : Π PAS récupérée"
              if g < 1.0 else "défense défaillante : Π récupérée")
    print("=" * 62, flush=True)
    print("RÉSUMÉ — VMA vue W_e·W_h (Table 9)", flush=True)
    print(f"  modèle : {model_subdir}", flush=True)
    print(f"  tokens : {n_tok} (sample={sample}) | blocs X : "
          f"{min(b + 1, n_blocks)}/{n_blocks}", flush=True)
    print(f"  TTRSR  : {g:.2f} %", flush=True)
    print(f"  ➜ {interp}", flush=True)
    print("=" * 62, flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=10800, scaledown_window=300)
def vma_combined_attack(
    model_subdir: str = "qwen3-8b-ft-h128-a1-h02",
    seed: int = 0,
    subset_size: int = 2000,
    sample: str = "uniform",
    block_rows: int = 2048,
    clear_ref: str = SRC_MODEL,
):
    """VMA COMBINÉE (Table 9) — la plus forte attaque mesurable : gate + W_e·W_h.

    Les deux vues testées sont combinées en un seul vote avant l'argmax :
      S_gate = cumul sur les 36 couches des similarités z-score (par bloc de
               8192 candidats — logique EXACTE de vma_product_full) ;
      S_weh  = similarités cosinus de la vue W_e·W_h (embed×head, V×V en
               streaming par blocs — logique de vma_weh_attack) ;
      vote   = zscore_ligne(S_gate) + zscore_ligne(S_weh) → argmax global.

    Le z-score final par ligne (sur les V candidats) est une transformation
    monotone : il ne change PAS l'argmax de chaque vue seule — les taux
    vote_gate / vote_weh rapportés ici doivent donc retomber sur les runs
    séparés (contrôle ~99,95 %/~100 %, défensif ~8,35 %/~0 %) — et il
    équilibre les échelles (gate cumule 36 couches, weh une seule) avant la
    somme 50/50.

    MESURÉ (2026-09-02, 2000 tokens uniformes) : contrôle base-h128-a0 →
    gate 99,95 % / We·Wh 100 % / combiné 100 % ; défensif
    qwen3-8b-ft-h128-a1-h02 (α_e=1,0/α_h=0,2) → gate 8,35 % / We·Wh 0 % /
    **combiné 6,55 %** — la vue gate seule reste la plus forte (We·Wh =
    bruit pur à ce réglage, il dilue le signal au lieu de l'ajouter).

    Sortie : vote_gate, vote_weh, vote_combined (TTRSR) + tokens récupérés
    du vote combiné.
    """
    import json
    import random
    import sys as _sys
    import time
    import urllib.request

    import torch
    from safetensors import safe_open

    _sys.path.insert(0, "/pkg/aloepri")
    from vma_product import _row_sort_chunked

    model_dir = os.path.join(MODELS_DIR, model_subdir)

    def _tensors_from_dir(directory, names):
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

    def _zscore_rows(s):
        return ((s - s.mean(dim=1, keepdim=True))
                / (s.std(dim=1, keepdim=True) + 1e-6))

    # --- poids obfusqués (volume) : embed + head ---
    obf_t = _tensors_from_dir(model_dir, ["model.embed_tokens.weight",
                                          "lm_head.weight"])
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    V = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]
    obf_embed = obf_t["model.embed_tokens.weight"]     # (V, d2) bf16
    obf_head = obf_t["lm_head.weight"]                  # (V, d2) bf16

    # --- poids clairs (référence publique HF) ---
    import json as _json
    with urllib.request.urlopen(
            f"https://huggingface.co/{clear_ref}/resolve/main/"
            "model.safetensors.index.json") as r:
        wm = _json.loads(r.read().decode())["weight_map"]
    from huggingface_hub import hf_hub_download

    def _hf_tensor(name):
        p = hf_hub_download(clear_ref, wm[name])
        with safe_open(p, framework="pt") as fp:
            return fp.get_tensor(name)

    clear_embed = _hf_tensor("model.embed_tokens.weight")
    clear_head = _hf_tensor("lm_head.weight")
    wnorm_fin = _hf_tensor("model.norm.weight")

    # --- permutation (seed) + échantillonnage (même logique que les runs) ---
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if sample == "uniform":
        test = torch.randperm(V)[:subset_size].to(dev)
    else:
        with open("/gepa_vocab_ids.txt") as f:
            gepa_ids = [int(l) for l in f if l.strip()]
        if sample == "gepa-tete":
            chosen = gepa_ids[:subset_size]
        elif sample == "gepa-strat":
            head_ids = gepa_ids[:100]
            rest = gepa_ids[100:]
            n_rest = subset_size - len(head_ids)
            idx_rest = torch.randperm(len(rest),
                                      generator=torch.Generator()
                                      .manual_seed(seed))[:max(0, n_rest)]
            chosen = head_ids + [rest[int(k)] for k in idx_rest.tolist()]
        else:
            raise ValueError(f"sample inconnu : {sample!r}")
        test = torch.tensor(chosen[:subset_size], device=dev)
    obf_rows = torch.tensor([perm[int(t)] for t in test.tolist()]).to(dev)
    n_tok = len(test)
    print(f"[combined] {n_tok} tokens testés (sample={sample}) sur "
          f"{model_subdir} — {n_layers} couches", flush=True)

    obf_embed = obf_embed.to(dev)
    obf_head = obf_head.to(dev)
    clear_embed = clear_embed.to(dev)

    # ============ VUE GATE (36 couches) ============
    t0 = time.time()
    S_gate = torch.zeros(n_tok, V, dtype=torch.float32, device=dev)
    for i in range(n_layers):
        obf_w = _tensors_from_dir(
            model_dir, [f"model.layers.{i}.mlp.gate_proj.weight"])
        clr = {}
        for n in (f"model.layers.{i}.mlp.gate_proj.weight",
                  f"model.layers.{i}.post_attention_layernorm.weight"):
            p = hf_hub_download(clear_ref, wm[n])
            with safe_open(p, framework="pt") as fp:
                clr[n] = fp.get_tensor(n).to(dev)
        wn = clr[f"model.layers.{i}.post_attention_layernorm.weight"]
        g_obf = obf_w[f"model.layers.{i}.mlp.gate_proj.weight"].to(dev).float()
        g_clair = (clr[f"model.layers.{i}.mlp.gate_proj.weight"]
                   * wn[None, :]).float()
        Y = obf_embed[obf_rows].float() @ g_obf.t()          # (N, inter)
        Y = _row_sort_chunked(Y)
        X = torch.empty(V, g_clair.shape[0],
                        dtype=torch.float32, device=dev)
        for c0 in range(0, V, 8192):
            c1 = min(c0 + 8192, V)
            X[c0:c1] = clear_embed[c0:c1].float() @ g_clair.t()
        X = _row_sort_chunked(X)
        q = torch.nn.functional.normalize(Y, dim=1)
        t = torch.nn.functional.normalize(X, dim=1)
        for c0 in range(0, V, 8192):
            c1 = min(c0 + 8192, V)
            sim = q @ t[c0:c1].t()                           # (N, 8192)
            sim = _zscore_rows(sim)                          # z-score par bloc
            S_gate[:, c0:c1] += sim
        rate = float((S_gate.argmax(dim=1) == test)
                     .float().mean().item())
        print(f"[combined] gate couche {i+1}/{n_layers} — "
              f"TTRSR cumulé {rate:.1%}", flush=True)
        del X, Y, q, t, sim, obf_w, clr
        torch.cuda.empty_cache()
    print(f"[combined] vue GATE terminée ({time.time()-t0:.0f}s) — "
          f"TTRSR(gate) {rate:.1%}", flush=True)

    # ============ VUE W_e·W_h ============
    t0 = time.time()
    Yw = obf_embed[obf_rows].float() @ obf_head.float().t()  # (N, V)
    del obf_embed, obf_head
    Yw = _row_sort_chunked(Yw)
    Yw = torch.nn.functional.normalize(Yw, dim=1)
    A = (clear_head.float() * wnorm_fin.float()[None, :]).t().to(dev)
    del clear_head
    S_weh = torch.zeros(n_tok, V, dtype=torch.float32, device=dev)
    n_blocks = (V + block_rows - 1) // block_rows
    for b in range(n_blocks):
        b0, b1 = b * block_rows, min((b + 1) * block_rows, V)
        Xb = clear_embed[b0:b1].float() @ A                   # (B, V)
        Xb = _row_sort_chunked(Xb)
        Xn = torch.nn.functional.normalize(Xb, dim=1)
        S_weh[:, b0:b1] = Yw @ Xn.t()                         # cos brut
        del Xb, Xn
        torch.cuda.empty_cache()
    print(f"[combined] vue W_e·W_h terminée ({time.time()-t0:.0f}s)", flush=True)

    # ============ VOTE COMBINÉ ============
    S_comb = _zscore_rows(S_gate) + _zscore_rows(S_weh)
    pred_gate = S_gate.argmax(dim=1)
    pred_weh = S_weh.argmax(dim=1)
    pred_comb = S_comb.argmax(dim=1)
    rate_gate = float((pred_gate == test).float().mean().item())
    rate_weh = float((pred_weh == test).float().mean().item())
    rate_comb = float((pred_comb == test).float().mean().item())

    result = {"modele": model_subdir, "n_tokens": n_tok, "sample": sample,
              "vote_gate": round(rate_gate, 4),
              "vote_weh": round(rate_weh, 4),
              "vote_combined": round(rate_comb, 4),
              "n_tokens_recuperes": int((pred_comb == test).sum().item()),
              "tokens_recuperes": test[pred_comb == test].tolist()}
    print("RESULTAT_VMA_COMBINED " + json.dumps(result), flush=True)

    g = rate_comb * 100
    interp = ("défense efficace : Π PAS récupérée"
              if g < 1.0 else "défense défaillante : Π récupérée")
    print("=" * 62, flush=True)
    print("RÉSUMÉ — VMA COMBINÉE gate + W_e·W_h (Table 9)", flush=True)
    print(f"  modèle : {model_subdir}", flush=True)
    print(f"  tokens : {n_tok} (sample={sample})", flush=True)
    print(f"  gate seul    : {rate_gate * 100:.2f} %", flush=True)
    print(f"  We·Wh seul   : {rate_weh * 100:.2f} %", flush=True)
    print(f"  COMBINÉ      : {g:.2f} %", flush=True)
    print(f"  ➜ {interp}", flush=True)
    print("=" * 62, flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=7200, scaledown_window=300)
def summ_generate(
    prompts: list[str],
    model_subdir: str = "qwen3-8b-ft-h128-a1-h02",
    seed: int = 0,
    max_new_tokens: int = 600,
    use_chat_template: bool = False,
    system: str = "",
):
    """Génération structurée (résumé wiki) sur le modèle OBFUSQUÉ du volume.

    Le modèle obfusqué génère par CONTINUATION ou par CHAT TEMPLATE.

    Par défaut (continuation) : le prompt est tokenisé sans tokens spéciaux.
    Avec `use_chat_template=True` (recommandé pour un modèle instruct
    Qwen3) : `apply_chat_template(enable_thinking=False)` — le template
    insère un bloc <think></think> vide qui désactive le raisonnement du
    Qwen3 chat (sinon il « pense » à voix haute en génération libre).
    Les ids (spéciaux compris) sont PERMUTÉS — la permutation préserve la
    sémantique du traitement ; la sortie est dépermutée puis décodée.

    Usage (benchmark résumé Wiki_LM) : prompts = contenu USER, `system` =
    consigne système, use_chat_template=True → résumés structurés sans
    raisonnement parasite.
    """
    import json
    import random
    import sys as _sys

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SRC_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODELS_DIR, model_subdir),
        dtype=torch.bfloat16).cuda().eval()

    V = model.config.vocab_size
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))
    unperm = {v: k for k, v in perm.items()}

    outs = []
    for text in prompts:
        if use_chat_template:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": text}]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=False, enable_thinking=False)
        else:
            ids = tok.encode(text, add_special_tokens=False)
        ids_in = [perm[i] for i in ids]
        input_ids = torch.tensor([ids_in]).cuda()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[0][input_ids.shape[1]:].tolist()
        gen = [unperm.get(i, i) for i in gen]
        outs.append(tok.decode(gen, skip_special_tokens=True).strip())
        print(f"[summ] {len(outs)}/{len(prompts)} — {len(outs[-1])} car.",
              flush=True)

    print("RESULTAT_SUMM_GENERATE " + json.dumps(
        {"n": len(outs), "longueurs": [len(o) for o in outs]},
        ensure_ascii=False), flush=True)
    return outs


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def precision_frwiki(
    seed: int = 0,
    n_files: int = 4,
    max_tokens: int = 4000,
    base_ref: str = "Qwen/Qwen3-8B",
    ft_ref: str = "qwen3-8b-ft-gepa",
    obf_ref: str = "qwen3-8b-ft-h128",
):
    """Précision sur un échantillon frwiki : perplexité + top-1 next-token.

    Comparaison base (HF) vs fine-tuné (9.1b) vs fine-tuné+obfusqué h>0
    (9.2). L'échantillon frwiki est embarqué dans l'image
    (`/frwiki_sample`, NDJSON wiki_XX). L'obfusqué reçoit les ids PERMUTÉS
    (perm régénérée par seed, comme vma_product_full) et ses logits sont
    dépermutés avant comparaison.
    """
    import json
    import os
    import random
    import sys as _sys

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- échantillon frwiki : NDJSON (une ligne = {"id","title","text",...})
    root = "/frwiki_sample"
    files = sorted(os.listdir(root))[:n_files]
    tok = AutoTokenizer.from_pretrained(base_ref)
    ids = []
    for n in files:
        with open(os.path.join(root, n), encoding="utf-8",
                  errors="ignore") as fh:
            for line in fh:
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if text:
                    ids += tok(text, add_special_tokens=False).input_ids
                if len(ids) > max_tokens:
                    break
        if len(ids) > max_tokens:
            break
    ids = torch.tensor(ids[:max_tokens])
    print(f"[frwiki] échantillon : {len(files)} fichiers → {ids.numel()} tokens",
          flush=True)

    # perm régénérée par seed (même convention que transform_chained /
    # vma_product_full : random.Random(seed).shuffle(range(V)))
    V = 151936  # vocabulaire Qwen3-8B (config vocabsize)
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))   # {clair: obfusqué}

    def _metrics(model_ref, perm=None):
        m = AutoModelForCausalLM.from_pretrained(
            model_ref, dtype=torch.bfloat16).cuda().eval()
        if perm:
            ids_in = torch.tensor([perm[int(t)] for t in ids.tolist()])
        else:
            ids_in = ids.clone()
        with torch.no_grad():
            logits = m(ids_in[None, :-1].cuda()).logits[0]   # (L-1, V)
        if perm:
            cols = torch.tensor([perm[t] for t in range(logits.shape[1])]).cuda()
            logits = logits[:, cols]
        loss = torch.nn.functional.cross_entropy(
            logits.float(), ids[1:].cuda()).item()
        top1 = float((logits.argmax(-1) == ids[1:].cuda()).float().mean().item())
        return {"perplexite": round(2 ** loss, 2), "top1_next_token": round(top1, 4)}

    # la base est chargée depuis HF (réf. publique) ; ft et obf depuis le
    # volume. ft_ref="" → pas de fine-tuné (colonne omise).
    result = {"base": _metrics(base_ref)}
    if ft_ref:
        result["ft"] = _metrics(os.path.join(MODELS_DIR, ft_ref))
    result["ft_obf_h128"] = _metrics(
        os.path.join(MODELS_DIR, obf_ref), perm=perm)
    print("RESULTAT_PRECISION_FRWIKI " + json.dumps(result), flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=3600, scaledown_window=300)
def precision_piaf(
    seed: int = 0,
    n_questions: int = 500,
    max_tokens: int = 64,
    base_ref: str = "Qwen/Qwen3-8B",
    ft_ref: str = "qwen3-8b-ft-gepa",
    obf_ref: str = "qwen3-8b-ft-h128-a1-h02",
):
    """Précision sur des QUESTIONS françaises (PiaF, Hub HF) : perplexité +
    top-1 next-token.

    Comparaison base (HF) vs fine-tuné (9.1b) vs fine-tuné+obfusqué h>0
    (9.2, α_e réglable via `obf_ref` — défaut α_e=1,0, le réglage défensif).
    Les questions PiaF (format SQuAD) sont téléchargées depuis le Hub HF
    (dataset `AgentPublic/piaf`) — questions SEULES, pas les contextes ni
    les réponses : c'est le texte « utilisateur » typique, court et
    interrogatif (structure très différente du texte courant frwiki).
    L'obfusqué reçoit les ids PERMUTÉS (perm régénérée par seed) et ses
    logits sont dépermutés avant comparaison.

    Les questions sont concaténées (séparées par eos) jusqu'à ~max_tokens ;
    la perplexité/top-1 portent donc sur le texte des questions.
    """
    import json
    import os
    import random
    import sys as _sys

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_ref)
    ds = load_dataset("AgentPublic/piaf", split="train")
    questions = [q for q in ds["question"] if q and isinstance(q, str)]
    random.Random(seed).shuffle(questions)
    questions = questions[:n_questions]

    # concaténation des questions (eos comme séparateur)
    ids = []
    for q in questions:
        ids += tok.encode(q, add_special_tokens=False)
        ids.append(tok.eos_token_id)
        if len(ids) > max_tokens:
            break
    ids = torch.tensor(ids[:max_tokens])
    print(f"[piaf] {len(questions)} questions → {ids.numel()} tokens",
          flush=True)

    V = 151936  # vocabulaire Qwen3-8B
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))   # {clair: obfusqué}

    def _metrics(model_ref, perm=None):
        m = AutoModelForCausalLM.from_pretrained(
            model_ref, dtype=torch.bfloat16).cuda().eval()
        if perm:
            ids_in = torch.tensor([perm[int(t)] for t in ids.tolist()])
        else:
            ids_in = ids.clone()
        with torch.no_grad():
            logits = m(ids_in[None, :-1].cuda()).logits[0]
        if perm:
            cols = torch.tensor([perm[t] for t in range(logits.shape[1])]).cuda()
            logits = logits[:, cols]
        loss = torch.nn.functional.cross_entropy(
            logits.float(), ids[1:].cuda()).item()
        top1 = float((logits.argmax(-1) == ids[1:].cuda()).float().mean().item())
        return {"perplexite": round(2 ** loss, 2), "top1_next_token": round(top1, 4)}

    result = {"base": _metrics(base_ref)}
    if ft_ref:
        result["ft"] = _metrics(os.path.join(MODELS_DIR, ft_ref))
    result["ft_obf_h128"] = _metrics(
        os.path.join(MODELS_DIR, obf_ref), perm=perm)
    print("RESULTAT_PRECISION_PIAF " + json.dumps(result), flush=True)
    return result


@app.function(image=TRANSFORM_IMAGE, volumes={MODELS_DIR: models_vol},
              gpu="A100-40GB", timeout=7200, scaledown_window=300,
              secrets=[modal.Secret.from_name("deepseek-api-key")])
def piaf_eval(
    seed: int = 0,
    n_pairs: int = 150,
    max_new_tokens: int = 80,
    base_ref: str = "Qwen/Qwen3-8B",
    obf_ref: str = "qwen3-8b-ft-h128-a1-h02",
    judge_model: str = "deepseek-chat",
):
    """Évaluation Q&A : génération sur couples PiaF puis jugement DeepSeek (1-5).

    Pipeline (votre demande) :
      1. GÉNÉRATION : `n_pairs` couples (question, contexte) de PiaF sont
         tokenisés ; le modèle OBFUSQUÉ (α_e=1,0) et la BASE (réf. publique)
         génèrent chacun une réponse. Pour l'obfusqué : ids PERMUTÉS envoyés,
         réponse dépermutée côté client (perm régénérée par seed, jamais sur
         Modal) — on sauve (question, contexte, réponse_obf, réponse_base).
      2. JUGEMENT : DeepSeek (API, secret `deepseek-api-key`) note chaque
         réponse de 1 à 5 en la comparant à la réponse de RÉFÉRENCE PiaF
         (critère objectif), avec le contexte.
      3. NOTE GLOBALE : moyenne des notes par modèle (obfusqué vs base).

    Sortie : JSON sauvegardé sur le volume `obfuscator-models/piaf_eval.json`
    + résumé lisible.
    """
    import json
    import os
    import random
    import sys as _sys
    import time
    import urllib.request

    import torch

    _sys.path.insert(0, "/pkg/aloepri")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_ref)
    ds = load_dataset("AgentPublic/piaf", split="train")
    pairs = []
    for q, c, a in zip(ds["question"], ds["context"], ds["answers"]):
        if q and c and a and isinstance(a, dict) and a.get("text"):
            pairs.append((q, c, a["text"][0]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    print(f"[piaf_eval] {len(pairs)} couples (question, contexte, réponse réf)",
          flush=True)

    # permutation (seed) — régénérée, jamais envoyée
    V = 151936
    rng_py = random.Random(seed)
    permuted = list(range(V))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(V), permuted))

    # modèles : base + obfusqué
    models = {}
    models["base"] = AutoModelForCausalLM.from_pretrained(
        base_ref, dtype=torch.bfloat16).cuda().eval()
    models["obf"] = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODELS_DIR, obf_ref), dtype=torch.bfloat16).cuda().eval()

    def _gen(model, text, perm=None):
        """Génère une réponse (greedy) ; perm → ids permutés puis dépermutation."""
        ids = tok.encode(text, add_special_tokens=False)
        if perm:
            ids_in = [perm[i] for i in ids]
        else:
            ids_in = ids
        input_ids = torch.tensor([ids_in]).cuda()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[0][input_ids.shape[1]:].tolist()
        if perm:
            unperm = {v: k for k, v in perm.items()}
            gen = [unperm.get(i, i) for i in gen]
        return tok.decode(gen, skip_special_tokens=True).strip()

    # --- 1. génération ---
    records = []
    for i, (q, c, ref) in enumerate(pairs):
        prompt = f"Question : {q}\nContexte : {c}\nRéponse :"
        r_base = _gen(models["base"], prompt)
        r_obf = _gen(models["obf"], prompt, perm=perm)
        records.append({"idx": i, "question": q, "contexte": c,
                        "reference": ref, "base": r_base, "obf": r_obf})
        if (i + 1) % 25 == 0:
            print(f"[gen] {i+1}/{len(pairs)} couples", flush=True)

    # --- 2. jugement DeepSeek (1-5) ---
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key, "DEEPSEEK_API_KEY manquant (secret deepseek-api-key)"

    def _judge(record, model_key):
        sys_prompt = (
            "Évaluateur strict. Compare la réponse à évaluer à la réponse de "
            "référence. Note 1 à 5 : 5=identique/exacte, 4=légèrement "
            "incomplète, 3=partiellement correcte, 2=vague ou très "
            "incomplète, 1=hors sujet ou vide. Réponds uniquement par "
            "l'entier.")
        user = (f"Question: {record['question']}\n"
                f"Contexte: {record['contexte'][:800]}\n"
                f"RÉFÉRENCE: {record['reference']}\n"
                f"À ÉVALUER ({model_key}): {record[model_key]}\n"
                f"Note (1-5) :")
        body = json.dumps({"model": judge_model,
                           "messages": [{"role": "system", "content": sys_prompt},
                                        {"role": "user", "content": user}],
                           "max_tokens": 5, "temperature": 0}).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        try:
            return max(1, min(5, int(content.strip())))
        except ValueError:
            return None   # réponse non parseable → non noté

    notes = {"base": [], "obf": []}
    for i, rec in enumerate(records):
        for mk in ("base", "obf"):
            n = _judge(rec, mk)
            if n is not None:
                notes[mk].append(n)
            rec[f"note_{mk}"] = n
        if (i + 1) % 25 == 0:
            print(f"[judge] {i+1}/{len(records)} couples notés "
                  f"(base moy {sum(notes['base'])/max(1,len(notes['base'])):.2f}, "
                  f"obf moy {sum(notes['obf'])/max(1,len(notes['obf'])):.2f})",
                  flush=True)
        time.sleep(0.3)   # respecter le rate-limit

    # --- 3. note globale ---
    result = {
        "n_pairs": len(pairs),
        "note_globale_base": round(sum(notes["base"]) / len(notes["base"]), 2),
        "note_globale_obf": round(sum(notes["obf"]) / len(notes["obf"]), 2),
        "n_notee_base": len(notes["base"]),
        "n_notee_obf": len(notes["obf"]),
        "detail": records,
    }
    out_path = os.path.join(MODELS_DIR, "piaf_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    models_vol.commit()
    print("RESULTAT_PIAF_EVAL " + json.dumps(
        {k: v for k, v in result.items() if k != "detail"}, ensure_ascii=False),
        flush=True)
    return result
