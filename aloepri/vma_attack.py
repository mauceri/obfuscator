"""VMA — Vocabulary-Matching Attack (Appendix D, Thomas et al. [25]).

Principe : la relation Y = Z1·X·Z2 relie la matrice claire X à la matrice
obfusquée Y. Le tri par ligne (RowSort) élimine la permutation de colonnes
Z2 ; l'appariement des lignes triées par plus proche voisin récupère la
permutation de lignes Z1 = Π (la permutation de vocabulaire).

Dans notre configuration (h=0, matrices clés désactivées), la vue la plus
directe est l'embedding : W̃_embed = Π·W⋆_embed (Z2 = I). On mesure la
récupération de Π par plus proche voisin (cosinus) des lignes obfusquées
contre la table claire — sur un sous-ensemble de tokens (la table complète
151936×d est trop grande pour une matrice de distances exhaustive).

Variante RowSort (fidèle au papier) : trier chaque ligne, puis apparier dans
l'espace trié — insensible à Z2 quand Z2 est une permutation.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def nearest_neighbor_rows(obf_rows, clear_table, chunk=2048):
    """Pour chaque ligne obfusquée, l'index clair le plus proche (cosinus).

    `obf_rows` : (N, d) float32 — lignes obfusquées à identifier.
    `clear_table` : (V, d) float32 — table claire.
    Retourne (N,) indices clairs prédits.
    """
    q = F.normalize(obf_rows, dim=1)
    t = F.normalize(clear_table, dim=1)
    preds = []
    for i in range(0, obf_rows.shape[0], chunk):
        sim = q[i:i + chunk] @ t.t()          # (chunk, V)
        preds.append(sim.argmax(dim=1))
    return torch.cat(preds)


@torch.no_grad()
def row_sort(rows):
    """Tri ascendant de chaque ligne (RowSort) — invariant aux permutations
    de colonnes."""
    return rows.sort(dim=1).values


def run_vma(obf_embed, clear_embed, perm, subset_size=2000, seed=0,
            use_row_sort=False):
    """Mesure la récupération de Π par VMA sur un sous-ensemble de tokens.

    `obf_embed` : table d'embedding obfusquée (V, d) — W̃_embed.
    `clear_embed` : table claire (V, d) — W_e.
    `perm` : dict {token clair -> index obfusqué} — la vraie permutation
    (l'évaluateur la connaît ; l'attaquant, non).
    Retourne (taux_récupération, nb_tokens_testés).
    """
    torch.manual_seed(seed)
    V = clear_embed.shape[0]
    # échantillon de tokens CLAIRS → leurs lignes obfusquées (index permutés)
    clear_tokens = torch.randperm(V)[:subset_size]
    obf_indices = torch.tensor([perm[int(t)] for t in clear_tokens.tolist()])

    obf_rows = obf_embed[obf_indices].float()
    if use_row_sort:
        obf_rows = row_sort(obf_rows)
        clr_all = row_sort(clear_embed.float())
    else:
        clr_all = clear_embed.float()
    # appariement contre la table claire COMPLÈTE (l'attaquant a le baseline)
    pred_clear = nearest_neighbor_rows(obf_rows, clr_all)
    correct = int((pred_clear == clear_tokens).sum().item())
    return correct / len(clear_tokens), len(clear_tokens)
