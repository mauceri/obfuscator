"""Attaques ISA (Internal State Attack) par descente de gradient.

Modèle de menace (AloePri arXiv 2603.01499, Appendix D.1 ; méthode
d'inversion du papier arXiv 2507.16372) : l'opérateur du serveur capture
l'état interne d'UNE requête réelle X1 (état caché d'une couche, ou scores
d'attention), puis optimise un candidat X2 par descente de gradient jusqu'à
ce que son état interne s'approche de l'état capturé — récupérant ainsi
l'entrée.

Implémentation (variante « soft tokens » du papier 2507.16372) :
- l'attaquant paramètre le candidat par des LOGITS PAR POSITION
  P ∈ R^{T×vocab} ; l'entrée du modèle est embeds = softmax(P/τ) @ W_embed
  (différentiable : le gradient remonte jusqu'à P à travers la table
  d'embedding) ;
- Adam + recuit de température τ (3 → 0,1) pour éviter les minima locaux ;
- à la fin, ids_récupérés = argmax(P) ;
- deux canaux : « hidden » (état caché d'une couche, MSE) et « attn »
  (pondérations d'attention d'une couche, MSE).

Métrique : taux de correspondance des IDs récupérés vs les IDs réellement
envoyés au modèle. Sur le modèle obfusqué, ces IDs sont les IDs PERMUTÉS —
les retrouver ne donne pas le texte (il faudrait la clé de permutation,
côté client). Le taux mesure donc la récupération de l'entrée « modèle »,
et la comparaison baseline vs obfusqué mesure l'effet de l'obfuscation.

Usage (module) :
    from .isa_attack import run_channel_attack, token_match_rate
"""
import argparse
import random

import torch


def capture_state(model, embeds, layer, channel="hidden"):
    """État interne (cible) du modèle pour `embeds` (T, d).

    channel="hidden" : état caché de sortie de la couche `layer`
    (bsz=1 → (T, hidden)).
    channel="attn" : pondérations d'attention de la couche `layer`
    (bsz=1, heads, T, T).
    """
    kw = {}
    if channel == "hidden":
        kw["output_hidden_states"] = True
    else:
        kw["output_attentions"] = True
    with torch.no_grad():
        out = model(inputs_embeds=embeds.to(model.dtype).unsqueeze(0), **kw)
    if channel == "hidden":
        state = out.hidden_states[layer][0]          # (T, hidden)
    else:
        state = out.attentions[layer][0]             # (heads, T, T)
    return state.detach()


def _soft_embeds(logits, embed_table, tau):
    """embeds (T, d) = softmax(P/τ) @ W — différentiable en P."""
    probs = torch.softmax(logits / tau, dim=-1)
    return probs @ embed_table


def attack_model(model, embed_table, target_state, seq_len, channel,
                 layer, steps=500, lr=0.05, tau_start=3.0, tau_end=0.1,
                 tau_steps=None, phase2_steps=200, seed=0, device="cuda"):
    """Descente de gradient sur des logits par position pour inverser l'état
    capturé `target_state`. Retourne (ids_récupérés (seq_len,), pertes).

    Deux phases (méthode du papier arXiv 2507.16372) :
    - phase 1 : optimisation douce avec recuit de température (3 → 0,1) ;
    - phase 2 : ré-initialisation des logits près de l'argmax de la phase 1
      + optimisation à température basse — corrige les choix discrets figés
      prématurément par le recuit.

    La perte est RELATIVE (MSE / variance de la cible) : les états cachés
    d'un vrai LLM ont des amplitudes de plusieurs ordres de grandeur, une MSE
    brute écraserait le gradient."""
    torch.manual_seed(seed)
    random.seed(seed)

    vocab = embed_table.shape[0]
    logits = torch.zeros(seq_len, vocab, device=device, requires_grad=True)
    # init léger hors de l'uniformité parfaite (évite le gradient nul)
    with torch.no_grad():
        logits += 0.05 * torch.randn_like(logits)

    target_var = target_state.float().pow(2).mean().item()
    if tau_steps is None:
        tau_steps = steps

    def _forward_step(lr_opt, tau):
        embeds = _soft_embeds(logits, embed_table, tau).to(model.dtype)
        kw = {}
        if channel == "hidden":
            kw["output_hidden_states"] = True
        else:
            kw["output_attentions"] = True
        out = model(inputs_embeds=embeds.unsqueeze(0), **kw)
        if channel == "hidden":
            state = out.hidden_states[layer][0]
        else:
            state = out.attentions[layer][0]
        loss = torch.nn.functional.mse_loss(
            state.float(), target_state.float()) / target_var
        lr_opt.zero_grad()
        loss.backward()
        lr_opt.step()
        return loss.item()

    opt = torch.optim.Adam([logits], lr=lr)
    losses = []
    for step in range(steps):
        tau = max(tau_end, tau_start * (tau_end / tau_start)
                  ** (step / max(1, tau_steps)))
        losses.append(_forward_step(opt, tau))
        if step % 50 == 0:
            print(f"  [isa] step {step}/{steps} loss {losses[-1]:.4f} "
                  f"(rel.) tau {tau:.3f}", flush=True)

    # --- phase 2 : affinage autour de l'argmax, température basse ---
    if phase2_steps and phase2_steps > 0:
        with torch.no_grad():
            onehot = torch.zeros_like(logits)
            onehot.scatter_(1, logits.argmax(-1, keepdim=True), 10.0)
            logits.data.copy_(onehot + 0.01 * torch.randn_like(logits))
        opt2 = torch.optim.Adam([logits], lr=lr * 0.5)
        for step in range(phase2_steps):
            losses.append(_forward_step(opt2, tau_end))
            if step % 50 == 0:
                print(f"  [isa] phase2 {step}/{phase2_steps} "
                      f"loss {losses[-1]:.4f} (rel.)", flush=True)

    with torch.no_grad():
        ids = logits.argmax(dim=-1)
    return ids, losses


def token_match_rate(pred_ids, true_ids):
    """Proportion d'IDs exactement récupérés (positions alignées)."""
    assert pred_ids.shape == true_ids.shape
    return (pred_ids == true_ids).float().mean().item()


def run_channel_attack(model, token_ids, channel, layer, embed_table=None,
                       steps=500, lr=0.05, seed=0, device="cuda"):
    """Attaque complète sur un modèle : capture de l'état de `token_ids`
    puis inversion. `token_ids` : IDs DÉJÀ dans l'espace du modèle (permutés
    pour l'obfusqué). Retourne (ids_récupérés, taux, pertes)."""
    if embed_table is None:
        embed_table = model.get_input_embeddings().weight
    # la table doit être en float32 pour le matmul softmax(P) @ W (le modèle
    # est en bf16) ; on n'a besoin que de la table, pas du reste du modèle
    embed_table = embed_table.detach().float()
    true_ids = torch.tensor(token_ids, device=device)
    with torch.no_grad():
        true_embeds = embed_table[true_ids]
    target = capture_state(model, true_embeds, layer, channel)
    pred, losses = attack_model(
        model, embed_table, target, len(token_ids), channel, layer,
        steps=steps, lr=lr, seed=seed, device=device,
    )
    return pred, token_match_rate(pred, true_ids), losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="répertoire modèle (HF)")
    ap.add_argument("--ids", required=True,
                    help="IDs (dans l'espace du modèle), séparés par des virgules")
    ap.add_argument("--channel", choices=["hidden", "attn"], default="hidden")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(device).eval()
    ids = [int(x) for x in args.ids.split(",")]
    pred, rate, _ = run_channel_attack(
        model, ids, args.channel, args.layer,
        steps=args.steps, lr=args.lr, seed=args.seed, device=device,
    )
    print(f"ids réels   : {ids}")
    print(f"ids attaqués: {pred.tolist()}")
    print(f"taux de correspondance : {rate:.1%}")


# ---------------------------------------------------------------------------
# Vocab-matching DISCRET (point 4 de la revue) — sans relaxation continue.
#
# La descente de gradient (attack_model) trouve des soft tokens dans
# l'enveloppe convexe du simplexe qui reproduisent l'état caché sans être le
# prompt — la loss converge (0,007) mais les ids ne matchent pas (0 %), ce
# qui ne permet PAS de conclure « canal hidden non informatif » (Thomas et
# al. : recherche discrète et autorégressive).
#
# Test discriminatif k-way : à chaque position, le vrai token est mélangé à
# k−1 leurres (tirage uniforme dans le vocabulaire). Le canal est informatif
# si le vrai token est identifié de façon fiable (taux → 100 %) ; s'il est
# sous-déterminé, le taux tend vers 1/k. Deux variantes :
#   - teacher_forcing=True : préfixe = VRAIS tokens (borne haute : isole la
#     capacité du canal à la couche `layer`) ;
#   - teacher_forcing=False : greedy autorégressif (préfixe = tokens prédits,
#     mesure la récupération réelle, propagation d'erreur incluse).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _state_at_position(model, embeds, layer, channel="hidden"):
    """État interne (T, *) du modèle pour `embeds` (B, T, d), dernière position
    de chaque ligne du batch. `embeds` est upcasté vers le dtype du modèle
    (le modèle est bf16, la table d'embedding d'attaque est float32)."""
    embeds = embeds.to(model.dtype)
    kw = {"output_hidden_states": True} if channel == "hidden" \
        else {"output_attentions": True}
    out = model(inputs_embeds=embeds, **kw)
    if channel == "hidden":
        return out.hidden_states[layer][:, -1]      # (B, hidden)
    attn = out.attentions[layer][:, :, -1, :]        # (B, heads, T) — dernière
    return attn.mean(dim=1)                          # position, moyenne têtes


def _state_distance(state, target_pos, target_var, metric="mse"):
    """Distance entre l'état d'un candidat (B, hidden) et la cible (hidden,)."""
    if metric == "mse":
        return (state.float() - target_pos.float()).pow(2).mean(dim=1) / target_var
    # cosinus (1 - cos) : insensible à l'échelle
    q = torch.nn.functional.normalize(state.float(), dim=1)
    t = torch.nn.functional.normalize(target_pos.float(), dim=0)
    return 1.0 - (q @ t)


def vocab_match_attack(model, embed_table, target_state, true_ids, layer,
                       k=64, channel="hidden", teacher_forcing=True,
                       metric="mse", batch=128, seed=0, device="cuda"):
    """Vocab-matching discret k-way (greedy autorégressif ou teacher-forced).

    `target_state` : (T, hidden) — état caché de la couche `layer` capturé
    sur l'entrée réelle (déjà dans l'espace du modèle, ids permutés).
    `true_ids` : (T,) — les ids réels (vérité terrain, pour l'évaluation ;
    l'attaquant ne les utilise PAS pour choisir, seulement pour mesurer).

    Retourne (ids_récupérés (T,), taux, taux_couche (T,)).
    """
    torch.manual_seed(seed)
    vocab = embed_table.shape[0]
    seq_len = len(true_ids)
    target_var = target_state.float().pow(2).mean().item()
    pred = torch.zeros(seq_len, dtype=torch.long, device=device)
    hit = torch.zeros(seq_len, dtype=torch.bool, device=device)

    for t in range(seq_len):
        # candidats : le vrai token + k−1 leurres, mélangés (ordre inconnu)
        prefix_ids = (true_ids[:t] if teacher_forcing else pred[:t])
        cand = torch.randint(0, vocab, (k - 1,), device=device)
        cand = torch.cat([true_ids[t:t + 1], cand])
        cand = cand[torch.randperm(k, device=device)]

        inp = torch.cat([
            prefix_ids.unsqueeze(0).expand(k, -1),
            cand.unsqueeze(1),
        ], dim=1)                                        # (k, t+1)
        embeds = embed_table[inp]
        dists = torch.empty(k, dtype=torch.float32, device=device)
        for c0 in range(0, k, batch):
            c1 = min(c0 + batch, k)
            st = _state_at_position(model, embeds[c0:c1], layer, channel)
            dists[c0:c1] = _state_distance(st, target_state[t],
                                           target_var, metric)
        chosen = int(dists.argmin().item())
        pred[t] = cand[chosen]
        hit[t] = (pred[t] == true_ids[t])
        print(f"  [vma] pos {t}: prédit {pred[t].item()} "
              f"({'✓' if hit[t] else '✗'}) — vrai {true_ids[t].item()}", flush=True)

    return pred, float(hit.float().mean().item()), hit


def run_vocab_match(model, token_ids, layer, k=64, channel="hidden",
                    teacher_forcing=True, metric="mse", seed=0, device="cuda"):
    """Attaque vocab-matching complète : capture de l'état puis discrimination
    k-way. `token_ids` : IDs déjà dans l'espace du modèle (permutés)."""
    embed_table = model.get_input_embeddings().weight.detach().float()
    true_ids = torch.tensor(token_ids, device=device)
    with torch.no_grad():
        true_embeds = embed_table[true_ids]
    target = capture_state(model, true_embeds, layer, channel)
    pred, rate, hit = vocab_match_attack(
        model, embed_table, target, true_ids, layer, k=k, channel=channel,
        teacher_forcing=teacher_forcing, metric=metric, seed=seed, device=device,
    )
    return pred, rate, hit
