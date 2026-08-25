"""Attaque par inversion des scores d'attention dans l'espace CLAIR.

Principe (variante ISA « clean-space ») : l'attaquant possède les poids
obfusqués (opérateur serveur) ET le modèle baseline public. Il capture les
scores d'attention d'une requête réelle sur le modèle obfusqué, puis optimise
dans le modèle CLAIR (poids gelés) un candidat soft-token
(`embeds = softmax(P/τ) @ W_embed`) pour reproduire ces scores. Au point de
convergence, `argmax(P)` donne directement les tokens EN CLAIR — la permutation
de vocabulaire est contournée (pas besoin de la clé).

La permutation de têtes τ (secrète) est gérée par appariement ICP (iterative
closest point) : à chaque pas, chaque tête claire est assignée à la tête
obfusquée la plus proche (assignation figée pour le pas, stop-gradient), ce
qui rend la perte invariante par relabeling des têtes.

Méthode : arXiv 2507.16372 (inversion par soft tokens, recuit + phase 2),
adaptée au canal attention dans l'espace clair.
"""
import random

import torch
import torch.nn.functional as F

try:
    from .isa_attack import capture_state
except ImportError:  # import top-level (conteneur Modal : sys.path vers aloepri/)
    from isa_attack import capture_state


def _soft_embeds(logits, embed_table, tau):
    """embeds (T, d) = softmax(P/τ) @ W — différentiable en P (soft tokens)."""
    probs = torch.softmax(logits / tau, dim=-1)
    return probs @ embed_table


def run_clean_space_inversion(clean_model, embed_table, target_scores,
                              seq_len, num_heads, steps=500, lr=0.05,
                              tau_start=3.0, tau_end=0.1, phase2_steps=0,
                              seed=0, device="cpu"):
    """Inverse les scores d'attention cibles dans le modèle CLAIR.

    `target_scores` : (num_heads, T, T) capturés sur le modèle obfusqué.
    `clean_model` : modèle baseline public, poids gelés (eval).
    Retourne (ids_clairs (T,), pertes).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    vocab = embed_table.shape[0]

    logits = torch.zeros(seq_len, vocab, device=device, requires_grad=True)
    with torch.no_grad():
        logits += 0.05 * torch.randn_like(logits)

    target = target_scores.float().detach()
    target_var = target.pow(2).mean().item()
    T = seq_len

    def _forward(opt, tau):
        embeds = _soft_embeds(logits, embed_table, tau).to(clean_model.dtype)
        # forward en mode grad : le gradient doit remonter jusqu'aux logits
        # (les poids du modèle clair sont gelés — seul `logits` est optimisé)
        out = clean_model(inputs_embeds=embeds.unsqueeze(0),
                          output_attentions=True)
        s_clean = out.attentions[0][0]  # (heads, T, T)
        # ICP : assignation des têtes claires aux têtes obfusquées les plus
        # proches (assignation figée pour ce pas — pas de gradient dedans).
        with torch.no_grad():
            d = torch.cdist(s_clean.float().view(num_heads, -1),
                            target.view(num_heads, -1))
            assignment = d.argmin(dim=1)  # (num_heads,)
        loss = F.mse_loss(s_clean.float(), target[assignment].to(s_clean.device))
        loss = loss / target_var
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    opt = torch.optim.Adam([logits], lr=lr)
    losses = []
    for step in range(steps):
        tau = max(tau_end, tau_start * (tau_end / tau_start) ** (step / max(1, steps)))
        losses.append(_forward(opt, tau))

    # phase 2 : affinage autour de l'argmax (comme isa_attack)
    if phase2_steps and phase2_steps > 0:
        with torch.no_grad():
            onehot = torch.zeros_like(logits)
            onehot.scatter_(1, logits.argmax(-1, keepdim=True), 10.0)
            logits.data.copy_(onehot + 0.01 * torch.randn_like(logits))
        opt2 = torch.optim.Adam([logits], lr=lr * 0.5)
        for _ in range(phase2_steps):
            losses.append(_forward(opt2, tau_end))

    with torch.no_grad():
        ids = logits.argmax(dim=-1)
    return ids, losses


def run_attention_inversion(obf_model, clean_model, obf_ids, layer=0,
                            steps=500, lr=0.05, seed=0, device="cpu"):
    """Attaque complète : capture des scores sur l'obfusqué, inversion claire.

    `obf_ids` : ids PERMUTÉS (l'entrée réelle du modèle obfusqué), calculés
    côté client avec les clés — l'attaquant les observe à l'inférence.
    Retourne (ids_clairs (T,), taux, pertes).
    """
    obf_embed = obf_model.get_input_embeddings().weight.detach().float()
    embeds = obf_embed[torch.tensor(obf_ids, device=device)]
    target = capture_state(obf_model, embeds, layer, channel="attn")
    num_heads = target.shape[0]
    embed_table = clean_model.get_input_embeddings().weight.detach().float()
    pred, losses = run_clean_space_inversion(
        clean_model, embed_table, target, len(obf_ids), num_heads,
        steps=steps, lr=lr, seed=seed, device=device,
    )
    return pred, losses
