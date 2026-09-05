"""
Obfuscation FFN : permutation de la dimension intermédiaire + scaling par
neurone, avec compensation inverse (papier §7.5 : "permutations and scaling
matrices").

silu(gate) * up est invariant à une permutation de la dimension intermédiaire
SI gate_proj et up_proj sont permutés identiquement en sortie, et down_proj
reçoit la même permutation en entrée : la somme sur l'axe intermédiaire ne
dépend pas de l'ordre de ses termes.

Pour le scaling : silu est non linéaire (silu(s·z) != s·silu(z) en général),
donc scaler gate_proj ne serait pas compensable exactement par un facteur sur
down_proj. Le scaling est donc limité à up_proj (facteur positif s_i par
neurone), qui n'entre pas dans silu :
    silu(gate) * (s_i · up) = s_i · (silu(gate) * up)
Ce facteur s_i, un par neurone intermédiaire, est annulé exactement par 1/s_i
sur la colonne correspondante de down_proj. gate_proj n'est donc que permuté,
jamais scalé.
"""
from dataclasses import dataclass
import random

import torch


@dataclass
class ObfuscatedFFN:
    gate_proj_obf: torch.Tensor
    up_proj_obf: torch.Tensor
    down_proj_obf: torch.Tensor


def obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed):
    intermediate_size, hidden_size = gate_proj.shape
    assert up_proj.shape == (intermediate_size, hidden_size)
    assert down_proj.shape == (hidden_size, intermediate_size)

    rng_py = random.Random(seed)
    perm = list(range(intermediate_size))
    rng_py.shuffle(perm)
    perm_index = torch.tensor(perm)

    gen = torch.Generator().manual_seed(seed)
    # scaling strictement positif pour rester inversible sans changer de signe
    scale = torch.exp(torch.randn(intermediate_size, generator=gen) * 0.1)

    gate_proj_obf = gate_proj[perm_index]
    up_proj_obf = up_proj[perm_index] * scale[perm_index].unsqueeze(1)
    down_proj_obf = down_proj[:, perm_index] / scale[perm_index].unsqueeze(0)

    return ObfuscatedFFN(gate_proj_obf, up_proj_obf, down_proj_obf)
