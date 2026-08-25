"""Tests de l'attaque VMA (Vocabulary-Matching Attack, Appendice D).

Verrouille trois comportements :
1. La permutation Π est récupérable à ~100 % sur une table claire sans bruit
   (α_e=0) par appariement cosinus direct — la relation W̃ = Π·W est
   parfaitement identifiable.
2. Le mécanisme RowSort du papier élimine bien une permutation de colonnes
   Z2 (Y = Z1·X·Z2) : l'appariement des lignes triées retrouve Z1 = Π.
3. Le bruit α_e aux niveaux du papier (α_e=0.3 comme α_e=1.0) ne protège PAS
   la vue embedding seule : le cosinus du vrai match (≈ 1/√(1+α²)) reste très
   supérieur au meilleur faux match sur une table gaussienne bien séparée.
   C'est le résultat de sécurité : dans notre configuration (h=0, matrices
   clés désactivées) l'attaquant ayant le modèle de base récupère Π
   directement depuis la table d'embedding obfusquée.
"""

import pytest
import torch

from ..vma_attack import nearest_neighbor_rows, row_sort, run_vma

torch.manual_seed(0)


def _random_table(V=512, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(V, d, generator=g)


def _random_perm(V, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm_ids = torch.randperm(V, generator=g).tolist()
    return {t: p for t, p in enumerate(perm_ids)}


def _obfuscated_like(clear, perm, alpha_e):
    """Reproduit `obfuscate_embedding` sans matrices clés :
    W̃[perm[t]] = W[t] + α·σ(W)·E[t]."""
    V, d = clear.shape
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(V, d, generator=g)
    star = clear + alpha_e * clear.std() * noise
    # Convention d'`obfuscate_embedding` : la ligne obfusquée p porte les
    # données du token clair unperm[p] ⟺ obf[perm[t]] = star[t].
    perm_idx = torch.tensor([perm[t] for t in range(V)])
    obf = torch.empty_like(star)
    obf[perm_idx] = star
    return obf


def test_vma_direct_recovers_noiseless():
    """α_e=0 : cosinus direct récupère ~100 % de Π (la vraie permutation)."""
    V = 512
    clear = _random_table(V=V)
    perm = _random_perm(V)
    obf = _obfuscated_like(clear, perm, alpha_e=0.0)
    rate, n = run_vma(obf, clear, perm, subset_size=200, seed=1)
    assert n == 200
    assert rate > 0.99


def test_row_sort_kills_column_permutation():
    """Mécanisme du papier : RowSort élimine Z2 dans Y = Z1·X·Z2."""
    V, d = 512, 32
    X = _random_table(V=V, d=d)
    Z1 = _random_perm(V, seed=1)          # permutation de lignes (= Π)
    Z2 = torch.randperm(d).tolist()       # permutation de colonnes
    idx1 = torch.tensor([Z1[t] for t in range(V)])
    Y = X[idx1][:, Z2]
    # RowSort des deux côtés puis appariement
    pred = nearest_neighbor_rows(row_sort(Y), row_sort(X))
    truth = torch.tensor([Z1[t] for t in range(V)])
    assert float((pred == truth).float().mean()) > 0.99


def test_embedding_noise_does_not_defend_direct_match():
    """α_e ∈ {0.3, 1.0} : la permutation reste récupérable à ~100 % sur une
    table gaussienne — le vrai match (cos ≈ 1/√(1+α²)) domine le meilleur
    faux match."""
    V = 512
    clear = _random_table(V=V)
    perm = _random_perm(V)
    for alpha in (0.3, 1.0):
        obf = _obfuscated_like(clear, perm, alpha_e=alpha)
        rate, _ = run_vma(obf, clear, perm, subset_size=200, seed=2)
        assert rate > 0.95, f"α={alpha} : récupération {rate:.3f}"


def test_noise_overwhelming_breaks_recovery():
    """À α très grand le bruit domine : la récupération s'effondre (bornes
    de l'attaque — le papier recommande α_e=1.0, pas plus)."""
    V = 512
    clear = _random_table(V=V)
    perm = _random_perm(V)
    obf = _obfuscated_like(clear, perm, alpha_e=4.0)
    rate, _ = run_vma(obf, clear, perm, subset_size=200, seed=3)
    assert rate < 0.5


def test_permutation_direction_contract():
    """Contrat de direction : `perm` = {clair -> obfusqué}, et la ligne
    obfusquée perm[t] porte bien les données du token clair t. Un échange
    de direction (unperm à la place de perm) doit casser la récupération —
    c'est le bug qui a donné 0 % en grandeur nature."""
    V = 512
    clear = _random_table(V=V)
    perm = _random_perm(V)
    unperm = {v: k for k, v in perm.items()}
    obf = _obfuscated_like(clear, perm, alpha_e=0.0)
    rate, _ = run_vma(obf, clear, unperm, subset_size=200, seed=4)
    assert rate < 0.1
