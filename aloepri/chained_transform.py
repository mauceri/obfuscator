"""Schéma complet AloePri (h>0) : chaînage global P̂/Q̂ sur la frontière hidden.

§5.4 : un couple global (P̂ ∈ R^{d×(d+2h)}, Q̂ ∈ R^{(d+2h)×d}, P̂·Q̂ = I_d)
traverse TOUTES les couches. Le modèle obfusqué a un `hidden_size` de d+2h :
les états cachés x̃ = x·P̂ circulent, et Q̂ = P̂⁻¹ (inverse à droite) est
absorbé dans l'unembedding.

Conventions (poids stockés (out, in), forward row-vecteur) :
  embed  : W̃_e = Π·(W*_e)·P̂                  (V, d+2h)
  head   : W̃_h = Π·(W*_h·Wnorm_fin)·Q̂ᵀ       (V, d+2h)
  q/k/v  : W̃ = (W·Wnorm_pre)·Q̂ᵀ              (head_out, d+2h)
  o      : W̃_o = P̂ᵀ·W_o                      (d+2h, head_in)
  gate/up: W̃ = (W·Wnorm_post)·Q̂ᵀ             (inter, d+2h)
  down   : W̃_down = P̂ᵀ·W_down                (d+2h, inter)

Norms (§5.2.5) : chaque RMSNorm est remplacée par une RMSNorm à poids scalaire
κ = E[‖xP̂‖/‖x‖]·√(d/(d+2h)) (x gaussien — hypothèse du papier), et sa
diagonale Wnorm est FUSIONNÉE dans la couche adjacente (q/k/v pour
input_layernorm, gate/up pour post_attention_layernorm, head pour la norme
finale) avant obfuscation. Le round-trip est APPROXIMATIF : l'erreur vient de
la variation par échantillon de ‖xP̂‖/‖x‖ autour de son espérance
(≈ CV ~ 1/√(h/2) : ~9 % à h=128, le réglage du papier).

L'obfuscation interne (permutation de têtes, Û_vo, bloc, FFN) se compose sur
les dimensions tête/intermédiaire, qui ne sont PAS sur la frontière P̂ — les
fonctions `obfuscate_attention_layer` / `obfuscate_ffn_layer` sont réutilisées
telles quelles, sur des poids déjà transformés par la frontière.

Sécurité : la vue directe de la VMA (embedding vs table claire) devient
impossible (dimensions d+2h ≠ d) ; seule reste la VMA produit (Table 9),
beaucoup plus faible — c'est la défense structurelle que le POC h=0 n'avait pas.
"""
import copy
import math
import random

import numpy as np
import torch

from .key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen
from .attention_obfuscation import obfuscate_attention_layer
from .ffn_obfuscation import obfuscate_ffn_layer


def estimate_kappa(p_hat, n=8192, seed=0):
    """κ = E[‖xP̂‖/‖x‖]·√(d/(d+2h)) — correction d'échelle §5.2.5.

    `p_hat` : (d, d+2h) — la matrice clé globale. L'hypothèse du papier est
    que l'entrée x de chaque norme suit une gaussienne : l'espérance est
    estimée par Monte Carlo sur N(0, I_d)."""
    P = torch.tensor(p_hat, dtype=torch.float64)
    d = P.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    ratio = (x @ P).norm(dim=1) / x.norm(dim=1)
    return float(ratio.mean().item() * math.sqrt(d / P.shape[1]))


def _vocab_permutation(vocab_size, seed):
    """Même tirage que `embedding_obfuscation.obfuscate_embedding` :
    random.Random(seed).shuffle — les clés restent valides entre h=0 et h>0."""
    rng_py = random.Random(seed)
    permuted = list(range(vocab_size))
    rng_py.shuffle(permuted)
    perm = dict(zip(range(vocab_size), permuted))
    unperm = {v: k for k, v in perm.items()}
    return perm, unperm


def calibrate_kappas(clear_model, p_hat, config, n_prompts=8, seq_len=32,
                     seed=0, dtype=torch.float64):
    """κ par couche, estimé sur les VRAIS états cachés (distribution réelle).

    Le papier suppose x gaussien (κ = E_Gauss[‖xP̂‖/‖x‖]·√(d/(d+2h))), mais
    les états réels ne le sont pas : l'espérance gaussienne introduit un
    biais d'échelle systématique (~1.2× mesuré sur Qwen3-0.6B) qui casse le
    round-trip. On estime donc E[‖xP̂‖/‖x‖] sur un passage du modèle CLAIR
    (même définition, distribution réelle) — un scalaire par norme, l'erreur
    résiduelle (variation par échantillon) reste celle du papier.

    Retourne une liste de κ alignée sur les hooks :
    [input_norm_0, post_attn_norm_0, input_norm_1, post_attn_norm_1, ...,
     norm_finale].
    """
    P = torch.tensor(p_hat, dtype=torch.float64)
    d = P.shape[0]
    scale = math.sqrt(d / P.shape[1])

    captures = []
    handles = []

    def _hook(name):
        def h(module, args, output):
            captures.append((name, args[0].detach().double()))
        return h

    for i, layer in enumerate(clear_model.model.layers):
        handles.append(layer.input_layernorm.register_forward_hook(
            _hook(f"in_{i}")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            _hook(f"post_{i}")))
    handles.append(clear_model.model.norm.register_forward_hook(
        _hook("final")))

    g = torch.Generator().manual_seed(seed)
    vocab = config.vocab_size
    try:
        for _ in range(n_prompts):
            ids = torch.randint(
                0, vocab, (1, seq_len), generator=g,
                dtype=torch.long, device=next(clear_model.parameters()).device)
            with torch.no_grad():
                clear_model(ids)
    finally:
        for hdl in handles:
            hdl.remove()

    kappas = {}
    counts = {}
    for name, x in captures:
        x = x.reshape(-1, x.shape[-1])
        r = (x @ P).norm(dim=1) / x.norm(dim=1)
        s = r.sum().item()
        kappas[name] = kappas.get(name, 0.0) + s
        counts[name] = counts.get(name, 0) + r.numel()
    order = ([f"in_{i}" for i in range(len(clear_model.model.layers))] +
             [f"post_{i}" for i in range(len(clear_model.model.layers))] +
             ["final"])
    return [kappas[n] / counts[n] * scale for n in order], order


def obfuscate_chained(clear_model, config, seed, alpha_e=1.0, alpha_h=0.2,
                      lam=0.3, h=128, beta=8, gamma=1e3, zeta=1e3,
                      rope_scaling=None, kappa_mode="empirical"):
    """Construit le modèle obfusqué complet (hidden_size = d+2h).

    Le modèle clair n'est PAS modifié — un nouveau modèle est construit avec
    `hidden_size = d + 2h` et ses poids remplis par la transformation chaînée.

    `kappa_mode` :
    - "empirical" (défaut) : κ par couche estimé sur les vrais états cachés
      (un passage de calibration sur le modèle clair) — corrige le biais
      d'échelle de l'hypothèse gaussienne du papier (mesuré ~1.2× sur
      Qwen3-0.6B) ;
    - "gaussian" : le κ unique du papier (E[‖xP̂‖/‖x‖] sur N(0, I_d)).

    Retourne (obf_model, keys) avec keys = {vocab_permutation,
    vocab_unpermute, seed} (même format que les artifacts POC).
    """
    d = config.hidden_size
    d2 = d + 2 * h

    rng = np.random.default_rng(seed)
    base = init_key_matrix(d, h, lam, rng)
    p_hat = key_mat_gen(base)       # (d, d+2h)
    q_hat = inv_key_mat_gen(base)   # (d+2h, d)
    P = torch.tensor(p_hat, dtype=torch.float64)
    Q = torch.tensor(q_hat, dtype=torch.float64)

    n_layers = len(clear_model.model.layers)
    if kappa_mode == "empirical":
        kappas, order = calibrate_kappas(clear_model, p_hat, config)
        kappa_of = {}
        for n, k in zip(order, kappas):
            if n == "final":
                kappa_final = k
            else:
                kind, idx = n.split("_")
                kappa_of[(kind, int(idx))] = k
    else:
        k_gauss = estimate_kappa(p_hat)
        kappa_of = {("in", i): k_gauss for i in range(n_layers)}
        kappa_of.update({("post", i): k_gauss for i in range(n_layers)})
        kappa_final = k_gauss

    new_cfg = copy.deepcopy(config)
    new_cfg.hidden_size = d2
    # Le chaînage rend embed (·P̂) et head (·Q̂ᵀ) distincts : le modèle
    # obfusqué est TOUJOURS délié, même si la source lie (ex. Qwen3-0.6B).
    new_cfg.tie_word_embeddings = False
    obf = type(clear_model)(new_cfg).eval()

    tied = (clear_model.get_input_embeddings().weight.data_ptr()
            == clear_model.get_output_embeddings().weight.data_ptr())
    if tied:
        print("[chained] embeddings liés dans la source : head délié dans "
              "l'obfusqué (W̃_e = W·P̂ vs W̃_h = W·Q̂ᵀ)")

    perm, unperm = _vocab_permutation(config.vocab_size, seed)
    inv_perm = torch.tensor([unperm[i] for i in range(config.vocab_size)])

    with torch.no_grad():
        # --- embed : bruit + permutation + P̂ ---
        W_e = clear_model.get_input_embeddings().weight.double()
        noise_e = torch.randn_like(
            W_e, generator=torch.Generator().manual_seed(seed))
        W_e_obf = (W_e + alpha_e * W_e.std() * noise_e)[inv_perm] @ P
        obf.get_input_embeddings().weight.copy_(W_e_obf)

        # --- head : bruit + permutation + fold Wnorm finale + Q̂ᵀ ---
        W_h = clear_model.get_output_embeddings().weight.double()
        wnorm_fin = clear_model.model.norm.weight.double()
        noise_h = torch.randn_like(
            W_h, generator=torch.Generator().manual_seed(seed + 1))
        W_h_obf = ((W_h + alpha_h * W_h.std() * noise_h)
                   * wnorm_fin[None, :])[inv_perm] @ Q.t()
        obf.get_output_embeddings().weight.copy_(W_h_obf)

        # --- norme finale → κ_final ---
        obf.model.norm.weight.copy_(torch.full((d2,), kappa_final))

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        d_head = config.head_dim
        for i, (cl, ol) in enumerate(zip(clear_model.model.layers,
                                         obf.model.layers)):
            wn1 = cl.input_layernorm.weight.double()
            wn2 = cl.post_attention_layernorm.weight.double()
            ol.input_layernorm.weight.copy_(
                torch.full((d2,), kappa_of[("in", i)]))
            ol.post_attention_layernorm.weight.copy_(
                torch.full((d2,), kappa_of[("post", i)]))

            # attention : frontière (Wnorm fusionné) puis obfuscation tête
            ac, ao = cl.self_attn, ol.self_attn
            layer_rope_scaling = (rope_scaling if rope_scaling is not None
                                  else not hasattr(ac, "q_norm"))
            layer_rope_rotation = not hasattr(ac, "q_norm")
            assert not hasattr(ac, "q_proj") or ac.q_proj.bias is None, (
                "biais d'attention non gérés en h>0 (Qwen3 : aucun)")
            w_q = (ac.q_proj.weight.double() * wn1[None, :]) @ Q.t()
            w_k = (ac.k_proj.weight.double() * wn1[None, :]) @ Q.t()
            w_v = (ac.v_proj.weight.double() * wn1[None, :]) @ Q.t()
            w_o = P.t() @ ac.o_proj.weight.double()
            obf_attn = obfuscate_attention_layer(
                w_q, w_k, w_v, w_o, num_heads=num_heads,
                num_kv_heads=num_kv_heads, d_head=d_head,
                beta=beta, gamma=gamma, zeta=zeta,
                seed=seed * 10000 + 2 * i,
                rope_layout="half", rope_scaling=layer_rope_scaling,
                rope_rotation=layer_rope_rotation,
            )
            ao.q_proj.weight.copy_(obf_attn.w_q_obf)
            ao.k_proj.weight.copy_(obf_attn.w_k_obf)
            ao.v_proj.weight.copy_(obf_attn.w_v_obf)
            ao.o_proj.weight.copy_(obf_attn.w_o_obf)
            # q_norm/k_norm : espace tête (hors frontière P̂) — inchangés
            ao.q_norm.weight.copy_(ac.q_norm.weight)
            ao.k_norm.weight.copy_(ac.k_norm.weight)

            # FFN : frontière (Wnorm fusionné) puis permutation/scaling
            gate = (cl.mlp.gate_proj.weight.double() * wn2[None, :]) @ Q.t()
            up = (cl.mlp.up_proj.weight.double() * wn2[None, :]) @ Q.t()
            down = P.t() @ cl.mlp.down_proj.weight.double()
            obf_ffn = obfuscate_ffn_layer(
                gate, up, down, seed=seed * 10000 + 2 * i + 1)
            ol.mlp.gate_proj.weight.copy_(obf_ffn.gate_proj_obf)
            ol.mlp.up_proj.weight.copy_(obf_ffn.up_proj_obf)
            ol.mlp.down_proj.weight.copy_(obf_ffn.down_proj_obf)

    keys = {"vocab_permutation": perm, "vocab_unpermute": unperm, "seed": seed}
    return obf, keys
