"""Obfuscation d'attention (Algorithme 2 + « Inter-head Permutation », papier
AloePri arXiv 2603.01499v2, page 9).

Algorithme 2, tel qu'imprimé (lignes 6-7) :

    6: W̃_k^{η(i)} = Q̂_k W_k^{η(i)} R̂_qk Ĥ_qk⁻¹ Ẑ_blockᵀ ,  W̃_v^{η(i)} = Q̂_v W_v^{η(i)} Û_vo
    7: W̃_q^{(i)}  = Q̂_q W_q^{(i)}  R̂_qk Ĥ_qk  Ẑ_block  ,  W̃_o^{(i)}  = Û_vo⁻¹ W_o^{(i)} P̂_o

Conventions
-----------
Le papier écrit les poids « ligne = dimension cachée » : W_q est (d, d_head) et
q = x·W_q. Les matrices clés (Q̂/P̂, d×d ici puisque h=0) agissent donc à
gauche, sur la frontière `hidden_size`, et R̂_qk/Ĥ_qk/Ẑ_block/Û_vo à droite,
sur la dimension de tête. HuggingFace stocke la transposée
(`w_q` = (num_heads·d_head, hidden)), donc toute multiplication à droite du
papier devient ici une multiplication à gauche par la transposée du facteur.

Trois points ont dû être tranchés pour que la reparamétrisation soit exacte.

1. Facteurs partagés par GROUPE, pas par tête. Le texte dit « We use
   Algorithm 2 to obfuscate the weights of a group of attention heads
   (W_q^{(i)}, W_k^{η(i)}, W_v^{η(i)}, W_o^{(i)}) » : un tirage par groupe GQA.
   C'est une nécessité mathématique, pas un détail : toutes les têtes Q d'un
   groupe font leur produit scalaire avec la MÊME tête K, donc elles doivent
   porter le même facteur droit, sans quoi rien ne s'annule. Idem pour Û_vo,
   partagé entre la tête V du groupe et les tranches de W_o des têtes Q du
   groupe.

2. Ẑ_block du côté K : Ẑ, pas Ẑᵀ. Le score obfusqué vaut
   q·A·Bᵀ·kᵀ avec A = R̂Ĥ Ẑ (ligne 7) et B le facteur de K (ligne 6) ; il faut
   donc A·Bᵀ = I, c'est-à-dire B = A⁻ᵀ = R̂⁻ᵀ Ĥ⁻ᵀ Ẑ⁻ᵀ = R̂ Ĥ⁻¹ Ẑ (R̂ orthogonale,
   Ĥ diagonale, Ẑ permutation). La ligne 6 imprimée donne bien R̂ et Ĥ⁻¹ dans
   cet ordre, mais Ẑᵀ au lieu de Ẑ⁻ᵀ = Ẑ : le produit devient Ẑ·Ẑ, qui ne vaut
   I que si Ẑ est une involution. C'est le cas tant que BlockPerm ne tire que
   des fenêtres de taille ≤ 2 (toutes les permutations de S₁/S₂ sont des
   involutions), mais plus du tout dès qu'une fenêtre contient un 3-cycle.
   Mesuré : à m_blocks=64 et β=8 (le cas du vrai modèle, d_head=128), 20/20
   tirages donnent ẐẐ ≠ I ; le round-trip avec Ẑᵀ produit alors une erreur de
   l'ordre de l'amplitude du signal (~1.0 pour une sortie d'amplitude ~1.5),
   contre ~1e-5 avec Ẑ. Les trois autres facteurs sont repris tels quels, seule
   l'orientation de Ẑ est corrigée.

3. Matrices clés Q̂_q/Q̂_k/Q̂_v/P̂_o non appliquées. Elles agissent sur la
   frontière `hidden_size` : dans le schéma complet du papier, Q̂ annule le P̂
   de la couche précédente (x̃ = x·P̂, P̂Q̂ = I) et P̂_o prépare la couche
   suivante. Ce POC assume explicitement de ne PAS chaîner les couches
   (`docs/superpowers/specs/2026-08-17-aloepri-poc-complet-design.md`, « chaque
   couche est obfusquée et vérifiée de façon indépendante, sans transformer la
   frontière hidden_size entre couches »). Sans chaînage, appliquer Q̂_q sur
   l'entrée réelle x donne x·Q̂_q·W_q ≠ x·W_q, et aucune autre matrice de la
   couche ne peut le compenser : Task 8 poserait ces poids dans un vrai modèle
   dont la sortie deviendrait du bruit. Elles sont donc omises ici — c'est la
   même limite que celle déjà actée pour le FFN, et la protection anti-ISA
   visée par le POC vient de la permutation tête/bloc, pas des matrices clés.

Ce qui reste (R̂_qk, Ĥ_qk, Ẑ_block, Û_vo, τ_kv, τ_group) s'annule intégralement
à l'intérieur de la couche : la sortie de l'attention est inchangée.

Conditionnement de Û_vo — DÉVIATION ASSUMÉE (décidée en Task 8). La ligne 4
du papier tire Û_vo ~ N(0, 1/d_head), mais une gaussienne carrée est souvent
très mal conditionnée : à d_head=128, une graine sur six produisait un
||W̃_o||_max de 1249 pour un ||W_o||_max d'origine de 0,45 (×2700), ce qui
détruit la sortie en bfloat16 (erreur relative mesurée jusqu'à 367 %) — le
dtype dans lequel le modèle est servi. Û_vo est donc tiré **orthogonal**
(Haar sur O(d_head), via QR d'une gaussienne + correction de signe) au lieu de
gaussien : c'est un affaiblissement du tirage (O(n) au lieu de GL(n)) mais
l'algèbre n'exige que Û_vo·Û_vo⁻¹ = I, la matrice reste un mélange dense et
aléatoire de la dimension de tête, et son conditionnement vaut exactement 1 —
donc aucune amplification des poids, et Û_vo⁻¹ = Û_voᵀ (exact, pas d'inversion
numérique). Sans cela, Task 9 mesurerait du bruit numérique et l'attribuerait
à l'obfuscation.

Biais q/k/v — Qwen2 met `bias=True` sur q_proj/k_proj/v_proj (contrairement à
Llama). Le biais s'ajoute APRÈS la projection, donc il doit subir exactement le
même facteur droit que le poids (q = x·W_qᵀ + b_q doit devenir (x·W_qᵀ + b_q)·A)
et suivre la même permutation inter-tête. Les paramètres `b_q`/`b_k`/`b_v` sont
optionnels (None pour une architecture sans biais) ; o_proj n'a pas de biais
dans Qwen2, et en aurait-il un qu'il ne serait pas concerné (Û_vo⁻¹ s'applique
à gauche, sur la dimension de tête, pas sur la sortie).

RoPE — ce module produit une reparamétrisation EXACTE de l'attention sans RoPE
(ce que vérifie le test). Les trois facteurs Q/K sont construits dans la
convention RoPE **entrelacée** du papier, où la paire tournée est (2i, 2i+1).
Dans cette convention : R̂_qk (rotation 2D dans le plan de la paire) et Ĥ_qk
(scalaire s_i I₂ sur la paire) commutent exactement avec la rotation RoPE et
restent donc exacts ; seul Ẑ_block, qui permute les fréquences à l'intérieur de
sa fenêtre, introduit une approximation — celle que le papier revendique
(« shuffling the RoPE's 2×2 blocks … within a limited window exerts minimal
impact on model accuracy »).

L'implémentation HF de Qwen2 n'utilise PAS cette convention : `rotate_half`
apparie (i, i + d_head/2). Sous cette convention, **les trois facteurs sont
faux, pas seulement R̂** :
- R̂ tourne le plan (2i, 2i+1), qui n'est pas un plan RoPE : elle ne commute
  plus avec la rotation RoPE ;
- Ĥ, diagonale, ne commute avec RoPE que si D[i] = D[i + d_head/2] ; or Ĥ est
  construite constante par paire (2i, 2i+1), donc cette égalité est fausse ;
- Ẑ permute des paires (2i, 2i+1) qui ne sont pas des paires RoPE du tout :
  il ne s'agit même plus du mélange de fréquences borné par une fenêtre décrit
  par le papier.

D'où le paramètre `rope_layout` (résolu en Task 8) :
- `"interleaved"` (défaut) : convention littérale du papier ;
- `"half"` : convention HF/Qwen2. Les facteurs sont alors conjugués par la
  permutation π entrelacé↔demi-vecteurs (πAπᵀ), ce qui les rend structurés
  selon les paires (i, i+d_head/2). L'invariance est préservée par
  construction, puisque (πAπᵀ)(πBπᵀ)ᵀ = π A Bᵀ πᵀ = π πᵀ = I, et R̂/Ĥ
  redeviennent exactement commutantes avec la rotation RoPE de HF (vérifié
  dans `tests/test_model_transform.py`). Ẑ reste, lui, l'approximation
  revendiquée par le papier — mesurée à 12-35 % d'erreur relative sur les
  scores à d_head=128, β=8, γ=1e3.

`rope_scaling` et les normes de tête Qwen3 — DÉVIATION ASSUMÉE (résolue pour
Qwen3, cf. `check_arch.py`). Qwen3 a introduit `q_norm`/`k_norm` : une RMSNorm
par tête appliquée à q/k immédiatement APRÈS la projection, AVANT RoPE
(`query_states = self.q_norm(self.q_proj(...))` dans `modeling_qwen3.py`).
Qwen2.5 n'en a pas. La RMSNorm de tête est non linéaire en l'échelle du
vecteur, donc elle ne commute avec un facteur A que si A préserve la norme
(rms(x·A) = rms(x)), c.-à-d. si A est orthogonale :
- R̂ (rotation 2D par paire) : orthogonale → commute exactement ;
- Ẑ_block (permutation par blocs, orthogonale) → commute exactement ;
- Ĥ (scaling diagonal par paire de fréquences) : **pas** orthogonale →
  rms(x·Ĥ) ≠ rms(x) → la RMSNorm de tête casse la reparamétrisation.
Avec Ĥ actif, le round-trip Qwen3 mesure une erreur de l'ordre de
l'amplitude du signal (mesuré ~20 % sur le modèle jouet) au lieu de ~1e-5.
`rope_scaling=False` remplace Ĥ par l'identité (tirage de la graine conservé
pour que le flux aléatoire reste identique entre les deux régimes) : A = B =
R̂·Ẑ reste orthogonale et le round-trip redevient exact. Le prix est la perte
du scaling par fréquence — composante mineure du schéma, et de toute façon
celle dont la commutation est structurellement impossible sous une RMSNorm
de tête. Le défaut reste `True` (comportement historique Qwen2) ;
`model_transform.py` choisit automatiquement `False` quand la couche expose
`q_norm`/`k_norm`.
"""
from dataclasses import dataclass
import random

import torch

from rope_transform import sample_rope_rotation, sample_rope_scaling
from block_perm import block_perm


@dataclass
class ObfuscatedAttention:
    w_q_obf: torch.Tensor
    w_k_obf: torch.Tensor
    w_v_obf: torch.Tensor
    w_o_obf: torch.Tensor
    b_q_obf: torch.Tensor = None
    b_k_obf: torch.Tensor = None
    b_v_obf: torch.Tensor = None


def _pi_conjugate(a, d_head):
    """πAπᵀ, où π réindexe les paires entrelacées (2i, 2i+1) en paires
    demi-vecteurs (i, i + d_head/2) — la convention `rotate_half` de HF."""
    idx = torch.cat([torch.arange(0, d_head, 2), torch.arange(1, d_head, 2)])
    return a[idx][:, idx]


def _random_orthogonal(n, gen):
    """Tirage Haar-uniforme sur O(n) : QR d'une gaussienne + correction de
    signe (sans quoi la loi n'est pas uniforme)."""
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    sign = torch.sign(torch.diagonal(r))
    sign[sign == 0] = 1.0
    return q * sign


def obfuscate_attention_layer(
    w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head, beta, gamma, zeta, seed,
    b_q=None, b_k=None, b_v=None, rope_layout="interleaved",
    rope_scaling=True,
):
    assert rope_layout in ("interleaved", "half"), rope_layout
    hidden_size = w_q.shape[1]
    assert w_q.shape == (num_heads * d_head, hidden_size)
    assert w_k.shape == (num_kv_heads * d_head, hidden_size)
    assert w_v.shape == (num_kv_heads * d_head, hidden_size)
    assert w_o.shape == (hidden_size, num_heads * d_head)
    assert num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads
    m_blocks = d_head // 2

    rng_py = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    q_heads = w_q.view(num_heads, d_head, hidden_size)
    k_heads = w_k.view(num_kv_heads, d_head, hidden_size)
    v_heads = w_v.view(num_kv_heads, d_head, hidden_size)
    o_heads = w_o.view(hidden_size, num_heads, d_head)

    q_obf = torch.zeros_like(q_heads)
    k_obf = torch.zeros_like(k_heads)
    v_obf = torch.zeros_like(v_heads)
    o_obf = torch.zeros_like(o_heads)

    # Biais (Qwen2 : bias=True sur q/k/v) — même facteur, même permutation.
    bias_heads = {}
    bias_obf = {}
    for name, b, n in (("q", b_q, num_heads), ("k", b_k, num_kv_heads),
                       ("v", b_v, num_kv_heads)):
        if b is None:
            continue
        assert b.shape == (n * d_head,), (name, b.shape)
        bias_heads[name] = b.view(n, d_head)
        bias_obf[name] = torch.zeros_like(bias_heads[name])

    # Permutation inter-tête : τ_kv déplace les têtes K/V (et donc les groupes
    # Q/O correspondants, sinon une tête Q n'attendrait plus la bonne tête K),
    # τ_group réordonne les têtes Q/O à l'intérieur de chaque groupe.
    tau_kv = list(range(num_kv_heads))
    rng_py.shuffle(tau_kv)
    tau_group = list(range(group_size))
    rng_py.shuffle(tau_group)

    for g in range(num_kv_heads):
        r_hat = sample_rope_rotation(d_head, seed=rng_py.randrange(2**31))
        # La graine est tirée AVANT la décision pour que le flux aléatoire
        # reste identique entre `rope_scaling=True/False` : la seule différence
        # entre les deux régimes est le facteur Ĥ lui-même (cf. docstring).
        h_seed = rng_py.randrange(2**31)
        h_hat = sample_rope_scaling(d_head, seed=h_seed) if rope_scaling else torch.eye(d_head)
        z_pairs = block_perm(beta, gamma, zeta, m_blocks, seed=rng_py.randrange(2**31))
        z_block = torch.kron(z_pairs, torch.eye(2))  # paires -> dimensions

        h_hat_inv = torch.diag(1.0 / torch.diagonal(h_hat))
        a_q_f32 = r_hat @ h_hat @ z_block  # ligne 7
        b_k_f32 = r_hat @ h_hat_inv @ z_block  # ligne 6, cf. point 2
        if rope_layout == "half":
            a_q_f32 = _pi_conjugate(a_q_f32, d_head)
            b_k_f32 = _pi_conjugate(b_k_f32, d_head)
        a_q = a_q_f32.to(w_q.dtype)
        b_k_mat = b_k_f32.to(w_k.dtype)

        # ligne 4 : le papier tire Û_vo ~ N(0, 1/d_head) ; ici orthogonale, cf.
        # « Conditionnement de Û_vo » en tête de module. Tirage en float32 puis
        # cast : bfloat16 dégraderait le tirage lui-même.
        u_vo_f32 = _random_orthogonal(d_head, gen)
        u_vo = u_vo_f32.to(w_v.dtype)
        u_vo_inv = u_vo_f32.T.to(w_o.dtype)  # orthogonale : inverse = transposée

        dst_g = tau_kv[g]
        k_obf[dst_g] = b_k_mat.T @ k_heads[g]
        v_obf[dst_g] = u_vo.T @ v_heads[g]
        if "k" in bias_obf:
            bias_obf["k"][dst_g] = b_k_mat.T @ bias_heads["k"][g]
        if "v" in bias_obf:
            bias_obf["v"][dst_g] = u_vo.T @ bias_heads["v"][g]
        for p in range(group_size):
            src_h = g * group_size + p
            dst_h = dst_g * group_size + tau_group[p]
            q_obf[dst_h] = a_q.T @ q_heads[src_h]
            o_obf[:, dst_h, :] = o_heads[:, src_h, :] @ u_vo_inv.T
            if "q" in bias_obf:
                bias_obf["q"][dst_h] = a_q.T @ bias_heads["q"][src_h]

    return ObfuscatedAttention(
        q_obf.reshape(num_heads * d_head, hidden_size),
        k_obf.reshape(num_kv_heads * d_head, hidden_size),
        v_obf.reshape(num_kv_heads * d_head, hidden_size),
        o_obf.reshape(hidden_size, num_heads * d_head),
        b_q_obf=None if "q" not in bias_obf else bias_obf["q"].reshape(-1),
        b_k_obf=None if "k" not in bias_obf else bias_obf["k"].reshape(-1),
        b_v_obf=None if "v" not in bias_obf else bias_obf["v"].reshape(-1),
    )
