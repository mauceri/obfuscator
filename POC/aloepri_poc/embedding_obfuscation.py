"""Obfuscation embedding/unembedding : bruit + permutation + matrices clés (papier §5.2.2).

`apply_key_matrices` — les matrices clés P̂_embed/Q̂_head agissent sur la
frontière `hidden_size`. Dans le schéma complet du papier (§5.4), c'est le même
changement de base P̂ qui traverse TOUTES les couches : x̃ = x·P̂ à la sortie de
l'embedding, chaque couche transformée en conséquence, puis Q̂ = P̂⁻¹ absorbé
dans l'unembedding. Ce POC a explicitement renoncé à ce chaînage (cf. la
décision « h=0 » du design et le point 3 de `attention_obfuscation.py`) : les
couches de décodeur ne sont PAS transformées sur cette frontière. Y appliquer
quand même P̂_embed livrerait au bloc 0 un état caché x·P̂ que plus rien ne
compense, et Q̂_head multiplierait un état caché jamais passé dans P̂ — le
modèle assemblé produirait du bruit.

`apply_key_matrices=False` (ce que `model_transform` utilise) ne garde donc que
le bruit et la permutation de vocabulaire, exactement comme l'attention ne
garde que ses facteurs intra-couche. Le défaut reste `True` pour préserver la
lecture littérale du papier vérifiée par les tests de la Task 3.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch

from key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen


@dataclass
class ObfuscatedEmbedding:
    w_embed_obf: torch.Tensor
    w_head_obf: torch.Tensor
    permutation: dict  # token clair -> token permuté
    unpermute: dict  # token permuté -> token clair


def obfuscate_embedding(w_embed, w_head, alpha_e, alpha_h, lam, h, seed,
                        apply_key_matrices=True):
    vocab_size, d = w_embed.shape
    assert w_head.shape == (vocab_size, d)

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    # Bruit gaussien : W* = W + alpha · σ(W) · bruit.
    # §5.2.2 « Noise Addition » : « The client samples noise matrices
    # E_embed ~ N(0, σ_e² I_n ⊗ I_d) et E_head ~ N(0, σ_h² I_d ⊗ I_n), where
    # σ_e, σ_h are the standard deviation of W_e, W_h ». α est donc un
    # coefficient RELATIF à l'écart-type de la matrice obfusquée, pas une
    # amplitude absolue : à α_e = 1.0 (défaut du papier) le bruit a la même
    # dispersion que le poids, quelle que soit l'échelle du modèle.
    noise_e = torch.randn(w_embed.shape, generator=torch.Generator().manual_seed(seed))
    noise_h = torch.randn(w_head.shape, generator=torch.Generator().manual_seed(seed + 1))
    w_embed_star = w_embed + alpha_e * w_embed.std() * noise_e
    w_head_star = w_head + alpha_h * w_head.std() * noise_h

    # permutation secrète du vocabulaire
    clear_ids = list(range(vocab_size))
    permuted_ids = list(range(vocab_size))
    rng_py.shuffle(permuted_ids)
    permutation = dict(zip(clear_ids, permuted_ids))
    unpermute = {v: k for k, v in permutation.items()}

    # Ligne `p` (un ID permuté) de la table obfusquée doit porter les
    # données du token clair `unpermute[p]` — c'est ce token-là que le
    # serveur doit reconnaître quand le client lui envoie l'ID permuté `p`.
    # D'où l'indexation par `unpermute` (== Π du papier), pas par
    # `permutation` (== Π⁻¹).
    inv_perm_index = torch.tensor([unpermute[i] for i in range(vocab_size)])

    if apply_key_matrices:
        # matrices clés (Algorithme 1) — une paire pour l'embedding, une pour le head
        base_embed = init_key_matrix(d, h, lam, rng_np)
        p_hat_embed = torch.tensor(key_mat_gen(base_embed), dtype=w_embed.dtype)

        base_head = init_key_matrix(d, h, lam, rng_np)
        q_hat_head = torch.tensor(inv_key_mat_gen(base_head), dtype=w_head.dtype)

        # W̃_embed = Π · W*_embed · P̂_embed
        w_embed_star = w_embed_star @ p_hat_embed
        # W̃_head = Q̂_head · W*_head · Πᵀ
        # (la sélection de lignes et la multiplication par la matrice clé
        # commutent : l'une agit sur l'axe vocab, l'autre sur l'axe d — l'ordre
        # n'a pas d'importance mathématiquement.)
        w_head_star = w_head_star @ q_hat_head.T

    w_embed_obf = w_embed_star[inv_perm_index]
    w_head_obf = w_head_star[inv_perm_index]

    return ObfuscatedEmbedding(w_embed_obf, w_head_obf, permutation, unpermute)
