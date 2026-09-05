import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from attention_obfuscation import obfuscate_attention_layer
from block_perm import block_perm


def naive_gqa_attention(x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head):
    seq_len, hidden_size = x.shape
    group_size = num_heads // num_kv_heads

    q = (x @ w_q.T).view(seq_len, num_heads, d_head)
    k = (x @ w_k.T).view(seq_len, num_kv_heads, d_head)
    v = (x @ w_v.T).view(seq_len, num_kv_heads, d_head)

    outputs = []
    for h in range(num_heads):
        kv_head = h // group_size
        scores = q[:, h] @ k[:, kv_head].T / (d_head**0.5)
        weights = torch.softmax(scores, dim=-1)
        outputs.append(weights @ v[:, kv_head])
    concat = torch.cat(outputs, dim=-1)
    return concat @ w_o.T


def _random_layer(hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed):
    torch.manual_seed(seed)
    return (
        torch.randn(num_heads * d_head, hidden_size) * 0.1,
        torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        torch.randn(hidden_size, num_heads * d_head) * 0.1,
        torch.randn(seq_len, hidden_size),
    )


def test_obfuscated_attention_preserves_output():
    hidden_size, num_heads, num_kv_heads, d_head = 32, 8, 2, 8
    seq_len = 5
    w_q, w_k, w_v, w_o, x = _random_layer(
        hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed=0
    )

    baseline = naive_gqa_attention(x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head)

    obf = obfuscate_attention_layer(
        w_q, w_k, w_v, w_o,
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
        beta=2, gamma=0.5, zeta=1e3, seed=0,
    )
    obf_output = naive_gqa_attention(
        x, obf.w_q_obf, obf.w_k_obf, obf.w_v_obf, obf.w_o_obf,
        num_heads, num_kv_heads, d_head,
    )

    torch.testing.assert_close(obf_output, baseline, atol=1e-3, rtol=1e-3)


def test_round_trip_holds_for_mha_mqa_and_other_seeds():
    """La composition doit tenir hors du seul cas GQA 8/2 du test principal :
    MHA (num_kv_heads == num_heads), MQA (num_kv_heads == 1), et plusieurs
    graines — une composition juste « par chance » sur un tirage ne survit pas
    à ce balayage.

    `d_head=32` (et non 8) est délibéré : avec `d_head=8` on n'a que
    `m_blocks=4` blocs RoPE, BlockPerm n'y tire que des fenêtres de taille ≤ 2
    et toutes les permutations de S₁/S₂ sont des involutions (ẐẐ = I). Dans ce
    régime Ẑ et Ẑᵀ sont interchangeables et le round-trip ne peut PAS
    distinguer l'orientation de Ẑ_block — alors que le vrai modèle
    (d_head=128 → m_blocks=64) tire des 3-cycles à tous les coups. Le test
    principal ci-dessus est donc aveugle à ce point précis ; celui-ci ne l'est
    pas (cf. l'assertion de régime plus bas)."""
    hidden_size, num_heads, d_head, seq_len = 64, 8, 32, 6
    beta, gamma, zeta = 8, 1e3, 1e3  # partagés avec l'assertion de régime

    # régime discriminant : au moins une partie des tirages BlockPerm doit
    # contenir un cycle de longueur ≥ 3, sinon ce test ne vaudrait pas mieux
    # que le précédent.
    def _z(s):
        return block_perm(beta, gamma, zeta, d_head // 2, seed=s)

    non_involutive = sum(
        bool(
            torch.linalg.matrix_norm(_z(s) @ _z(s) - torch.eye(d_head // 2)) > 1e-6
        )
        for s in range(20)
    )
    assert non_involutive >= 10, (
        "paramètres non discriminants : BlockPerm ne tire que des involutions"
    )

    for num_kv_heads in (8, 4, 1):
        for seed in (1, 2, 3):
            w_q, w_k, w_v, w_o, x = _random_layer(
                hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed
            )
            baseline = naive_gqa_attention(
                x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head
            )
            obf = obfuscate_attention_layer(
                w_q, w_k, w_v, w_o,
                num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
                beta=beta, gamma=gamma, zeta=zeta, seed=seed,
            )
            got = naive_gqa_attention(
                x, obf.w_q_obf, obf.w_k_obf, obf.w_v_obf, obf.w_o_obf,
                num_heads, num_kv_heads, d_head,
            )
            torch.testing.assert_close(
                got, baseline, atol=1e-3, rtol=1e-3,
                msg=f"round-trip cassé pour num_kv_heads={num_kv_heads}, seed={seed}",
            )


def _row_space_projector(m):
    """Projecteur orthogonal sur l'espace engendré par les lignes de `m`."""
    basis = torch.linalg.qr(m.T).Q  # (hidden, d_head)
    return basis @ basis.T


def _recovered_factor(clear_head, obf_head):
    """Facteur intra-tête A tel que `obf_head = Aᵀ · clear_head`.

    Récupérable sans connaître les clés dès que `hidden > d_head` : les lignes
    de `clear_head` sont alors libres, donc A est déterminé exactement."""
    return torch.linalg.lstsq(clear_head.T, obf_head.T).solution


def test_heads_are_actually_permuted_and_transformed():
    """Le round-trip seul ne distingue pas une implémentation qui ne ferait
    rien : il passerait aussi avec l'identité. Ce test vérifie que les poids
    changent réellement ET que la permutation inter-tête est appliquée de
    façon cohérente.

    Chaque tête est transformée par multiplication à gauche par une matrice
    inversible (côté d_head), ce qui laisse l'espace des lignes (sous-espace
    de dimension d_head de R^hidden) invariant. On peut donc retrouver la
    permutation appliquée en appariant les espaces de lignes, sans connaître
    les clés — et vérifier que les têtes Q suivent bien leur groupe K/V (sans
    quoi le calcul serait faux, ce que le round-trip attraperait, mais aussi
    qu'elles sont bien déplacées, ce qu'il n'attraperait pas)."""
    # d_head=32 (m_blocks=16) et non 8 : à 4 blocs seulement, tous d'indice
    # « haute fréquence », BlockPerm ne tire que des fenêtres singleton et
    # rend l'identité — Ẑ_block n'obfusquerait alors rien du tout et
    # l'assertion correspondante ci-dessous serait invérifiable.
    # num_kv_heads=4 (et non 2) pour que τ_kv = identité ne soit pas un simple
    # pile ou face : l'assertion « τ_kv permute vraiment » aurait une chance
    # sur deux d'être vide avec 2 têtes K/V.
    hidden_size, num_heads, num_kv_heads, d_head = 64, 8, 4, 32
    group_size = num_heads // num_kv_heads
    w_q, w_k, w_v, w_o, _ = _random_layer(
        hidden_size, num_heads, num_kv_heads, d_head, 4, seed=7
    )
    obf = obfuscate_attention_layer(
        w_q, w_k, w_v, w_o,
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
        beta=8, gamma=1e3, zeta=1e3, seed=7,
    )

    assert not torch.allclose(obf.w_q_obf, w_q)
    assert not torch.allclose(obf.w_k_obf, w_k)
    assert not torch.allclose(obf.w_v_obf, w_v)
    assert not torch.allclose(obf.w_o_obf, w_o)

    def match(clear, obfusc, n_heads):
        clear_heads = clear.view(n_heads, d_head, -1)
        obf_heads = obfusc.view(n_heads, d_head, -1)
        mapping = []
        for src in range(n_heads):
            p_src = _row_space_projector(clear_heads[src])
            dists = [
                torch.linalg.matrix_norm(p_src - _row_space_projector(obf_heads[dst]))
                for dst in range(n_heads)
            ]
            best = int(torch.tensor(dists).argmin())
            assert dists[best] < 1e-3, "aucune tête obfusquée ne correspond"
            mapping.append(best)
        return mapping

    tau_kv = match(w_k, obf.w_k_obf, num_kv_heads)
    assert sorted(tau_kv) == list(range(num_kv_heads))
    assert tau_kv != list(range(num_kv_heads)), "les têtes K/V ne sont pas permutées"
    assert match(w_v, obf.w_v_obf, num_kv_heads) == tau_kv, "K et V doivent suivre τ_kv"

    sigma = match(w_q, obf.w_q_obf, num_heads)
    assert sorted(sigma) == list(range(num_heads))
    assert sigma != list(range(num_heads)), "les têtes Q ne sont pas permutées"
    for h, dst in enumerate(sigma):
        assert dst // group_size == tau_kv[h // group_size], (
            "une tête Q a changé de groupe sans suivre sa tête K/V"
        )

    # τ_group : réordonnancement des têtes Q À L'INTÉRIEUR du groupe. Sans
    # cette assertion, τ_group dégénéré en identité passerait inaperçu, le
    # déplacement des groupes par τ_kv suffisant à rendre `sigma` non trivial.
    # Le papier ne tire qu'un seul τ_group ~ S_{m/m_kv}, donc le même
    # réordonnancement doit valoir pour tous les groupes.
    intra_groupe = [
        [sigma[g * group_size + p] % group_size for p in range(group_size)]
        for g in range(num_kv_heads)
    ]
    assert all(w == intra_groupe[0] for w in intra_groupe), (
        "τ_group devrait être unique pour tous les groupes"
    )
    assert intra_groupe[0] != list(range(group_size)), (
        "τ_group ne réordonne pas les têtes Q du groupe"
    )

    # Aucun facteur intra-tête ne doit dégénérer en identité : chaque tête
    # déplacée doit AUSSI être transformée. Les `allclose` globaux ci-dessus ne
    # le garantissent pas (ils sont satisfaits dès que la permutation bouge
    # quelque chose) ; avec Û_vo = I, par exemple, le round-trip reste vert à
    # 6.6e-07 près.
    for clear, obfusc, mapping, n_heads in (
        (w_q, obf.w_q_obf, sigma, num_heads),
        (w_k, obf.w_k_obf, tau_kv, num_kv_heads),
        (w_v, obf.w_v_obf, tau_kv, num_kv_heads),
    ):
        clear_heads = clear.view(n_heads, d_head, -1)
        obf_heads = obfusc.view(n_heads, d_head, -1)
        for src, dst in enumerate(mapping):
            assert not torch.allclose(obf_heads[dst], clear_heads[src], atol=1e-6), (
                f"tête {src} déplacée en {dst} mais laissée telle quelle"
            )

    # R̂_qk effectivement appliquée. Ĥ (diagonale) et Ẑ (permutation de paires)
    # ne laissent qu'UN coefficient non nul par ligne du facteur intra-tête ;
    # seule la rotation R̂ en met deux, en mélangeant les deux dimensions de
    # chaque paire RoPE. Sans cette vérification, R̂ = I passerait tout le
    # reste (round-trip vert à 1.6e-05 près).
    hors_bloc = torch.ones(d_head, d_head)  # masque : hors des blocs 2×2 diagonaux
    for i in range(d_head // 2):
        hors_bloc[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 0
    z_deplace_un_bloc = False

    for clear, obfusc, mapping, n_heads in (
        (w_q, obf.w_q_obf, sigma, num_heads),
        (w_k, obf.w_k_obf, tau_kv, num_kv_heads),
    ):
        clear_heads = clear.view(n_heads, d_head, -1)
        obf_heads = obfusc.view(n_heads, d_head, -1)
        for src, dst in enumerate(mapping):
            factor = _recovered_factor(clear_heads[src], obf_heads[dst])
            nnz_par_ligne = (factor.abs() > 1e-6).sum(dim=1)
            assert int(nnz_par_ligne.min()) >= 2, (
                "le facteur intra-tête ne mélange pas les paires RoPE : R̂_qk "
                "est-elle dégénérée en identité ?"
            )
            # Ĥ_qk : sans elle le facteur serait R̂·Ẑ, donc orthogonal.
            ecart_orthogonal = torch.linalg.matrix_norm(
                factor.T @ factor - torch.eye(d_head)
            )
            assert float(ecart_orthogonal) > 1e-3, (
                "facteur intra-tête orthogonal : Ĥ_qk est-elle dégénérée en "
                "identité ?"
            )
            # Ẑ_block : s'il permute réellement des paires, le facteur a des
            # coefficients hors des blocs 2×2 diagonaux. Un tirage peut rendre
            # l'identité, d'où la vérification globale (au moins un groupe).
            z_deplace_un_bloc |= bool((factor * hors_bloc).abs().max() > 1e-6)

    assert z_deplace_un_bloc, "Ẑ_block ne permute aucune paire RoPE"
