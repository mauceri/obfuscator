"""VMA PRODUIT (Table 9 du papier) — la variante complète de l'Appendice D.

L'attaque directe (embedding vs table claire) est structurellement impossible
en h>0 (dimensions d+2h ≠ d). Le papier contourne via les PRODUITS de poids :
les matrices clés s'annulent (P̂·Q̂ = I) et le produit retombe dans un espace
comparable. Vue utilisée ici (la plus propre pour Qwen3 dense) :

    Y = W̃_embed · W̃_gateᵀ  =  Π · (W*_e · (W_gate·Wnorm)ᵀ) · Ẑ_ffn

- Π : permutation de vocabulaire (récupérée par appariement de lignes) ;
- Ẑ_ffn : permutation de la dimension intermédiaire (facteur FFN) — éliminée
  par RowSort (tri ascendant de chaque ligne) ;
- le bruit d'embedding α_e·σ·E est la SEULE erreur qui survit (corrélée entre
  les couches : le vote multi-couches ne peut pas la moyenner).

Mécanique (mêmes conventions que `vma_attack.py`) :
1. Y_sub = W̃_embed[lignes test] @ W̃_gateᵀ      (N × inter)
2. X_clair = W_e @ (W_gate·Wnorm)ᵀ               (V × inter, complet)
3. RowSort(Y_sub) et RowSort(X_clair)
4. plus proche voisin (cosinus, chunké) → prédiction Π par couche
5. vote majoritaire sur les couches → prédiction finale

La Table 3 du papier mesure TTRSR contre VMA produit sur PUPA : Qwen3-14B
25,05 %, Qwen3-32B 19,64 % (les plus hauts de tous les modèles) — à α_e=1.0.
Notre déploiement (α_e=0.3) est potentiellement plus vulnérable : à mesurer.
"""
import torch
import torch.nn.functional as F

try:
    from .vma_attack import nearest_neighbor_rows, row_sort
except ImportError:  # conteneur Modal : import top-level
    from vma_attack import nearest_neighbor_rows, row_sort


@torch.no_grad()
def product_rows(embed, gate, wnorm=None, rows=None, chunk=2048,
                 dtype=torch.float32):
    """Lignes du produit W_embed · (W_gate·Wnorm)ᵀ.

    `embed` : (V, d) — table d'embedding.
    `gate` : (inter, d) — poids gate (la diagonale Wnorm y est DÉJÀ pliée
    pour le côté clair ; l'obfusqué est déjà complètement transformé).
    `wnorm` : (d,) facultatif — plié ici si le gate clair ne l'a pas.
    `rows` : indices de lignes à calculer (None = toutes) — la table complète
    est construite par blocs pour limiter la mémoire.
    Retourne (nb_lignes, inter) en `dtype`.
    """
    M = gate if wnorm is None else gate * wnorm[None, :]
    Mt = M.t().to(dtype)                        # (d, inter)
    if rows is None:
        out = torch.empty(embed.shape[0], Mt.shape[1], dtype=dtype,
                          device=embed.device)
        src = embed.to(dtype)
        for c0 in range(0, src.shape[0], chunk):
            c1 = min(c0 + chunk, src.shape[0])
            out[c0:c1] = src[c0:c1] @ Mt
        return out
    src = embed[rows].to(dtype)
    return src @ Mt


@torch.no_grad()
def _row_sort_chunked(x, chunk_rows=16384):
    """RowSort par blocs : le sort GPU upcast bf16→fp32 (valeurs + indices,
    ~2× la table) — trier la table complète en une fois a fait un OOM CUDA
    (13,9 GiB sur les 40 de l'A100). Chaque bloc est trié séparément."""
    out = torch.empty_like(x)
    for c0 in range(0, x.shape[0], chunk_rows):
        c1 = min(c0 + chunk_rows, x.shape[0])
        out[c0:c1] = x[c0:c1].sort(dim=1).values
    return out


def run_vma_product(obf_embed, obf_gates, clear_embed, clear_gates,
                    wnorm_list, perm, subset_size=2000, seed=0,
                    chunk=2048, dtype=torch.float32):
    """VMA produit multi-couches (vote majoritaire).

    `obf_embed` : (V, d2) table obfusquée (W̃_e).
    `obf_gates` : liste de (inter, d2) — W̃_gate par couche (déjà transformées).
    `clear_embed` : (V, d) table claire (W_e).
    `clear_gates` : liste de (inter, d) — W_gate clairs (SANS Wnorm plié).
    `wnorm_list` : liste de (d,) — poids post_attention_layernorm par couche.
    `perm` : dict {clair -> obfusqué} — vérité terrain (l'évaluateur).
    Retourne dict {taux_vote, n, taux_par_couche, votes_par_token}.
    """
    V = clear_embed.shape[0]
    torch.manual_seed(seed)
    clear_tokens = torch.randperm(V)[:subset_size].to(obf_embed.device)
    obf_rows_idx = torch.tensor(
        [perm[int(t)] for t in clear_tokens.tolist()]).to(obf_embed.device)

    preds = []
    for i, (g_obf, g_clair, wn) in enumerate(zip(obf_gates, clear_gates,
                                                 wnorm_list)):
        # Y_sub : lignes test du produit obfusqué (W̃_e · W̃_gateᵀ)
        Y = obf_embed[obf_rows_idx].to(dtype) @ g_obf.to(dtype).t()
        Y = row_sort(Y)
        # X_clair : table complète du produit clair (W_e · (W_gate·Wnorm)ᵀ)
        X = product_rows(clear_embed, g_clair, wn, rows=None,
                         chunk=chunk, dtype=dtype)
        X = _row_sort_chunked(X)
        pred = nearest_neighbor_rows(Y, X, chunk=chunk)
        rate = float((pred == clear_tokens).float().mean().item())
        preds.append(pred)
        print(f"[vma_produit] couche {i}: TTRSR = {rate:.1%} "
              f"({len(clear_tokens)} tokens)", flush=True)
        is_cuda = X.device.type == "cuda"
        del X
        if is_cuda:
            torch.cuda.empty_cache()

    votes = torch.stack(preds)
    final = votes.mode(dim=0).values
    vote_rate = float((final == clear_tokens).float().mean().item())
    return {
        "taux_vote": vote_rate,
        "taux_par_couche": [float((p == clear_tokens).float().mean().item())
                            for p in preds],
        "n": len(clear_tokens),
        "n_couches": len(preds),
    }
