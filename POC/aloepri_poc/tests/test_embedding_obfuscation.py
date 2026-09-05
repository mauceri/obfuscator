import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from embedding_obfuscation import obfuscate_embedding


def test_obfuscated_embedding_round_trip_preserves_output_up_to_noise():
    torch.manual_seed(0)
    vocab_size, d = 50, 16
    w_embed = torch.randn(vocab_size, d)
    w_head = torch.randn(vocab_size, d)

    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=0.0, alpha_h=0.0, lam=0.3, h=0, seed=0
    )

    assert result.w_embed_obf.shape == w_embed.shape
    assert result.w_head_obf.shape == w_head.shape
    # sans bruit (alpha=0), l'obfuscation est une reparamétrisation exacte :
    # regarder un token clair t, le convertir en ID permuté, lire la ligne
    # correspondante de l'embedding obfusqué, et vérifier qu'elle reproduit
    # (via la matrice clé) la ligne originale.
    clear_token = 5
    permuted_token = result.permutation[clear_token]
    assert result.unpermute[permuted_token] == clear_token


def test_permutation_is_a_bijection_over_the_vocabulary():
    torch.manual_seed(1)
    vocab_size, d = 30, 8
    w_embed = torch.randn(vocab_size, d)
    w_head = torch.randn(vocab_size, d)
    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=1.0, alpha_h=0.2, lam=0.3, h=0, seed=1
    )
    assert sorted(result.permutation.values()) == list(range(vocab_size))
    assert sorted(result.unpermute.values()) == list(range(vocab_size))


def test_noise_amplitude_is_relative_to_the_standard_deviation_of_the_weights():
    """§5.2.2 : « E_embed ~ N(0, σ_e² …), where σ_e, σ_h are the standard
    deviation of W_e, W_h ». α est donc un coefficient RELATIF. Sans le facteur
    σ(W), α_e = 1.0 ajouterait un bruit d'écart-type 1 à une table dont les
    coefficients valent ~0,02 — un rapport bruit/signal de 50 au lieu de 1, qui
    détruirait le modèle. Ce test échoue si le facteur σ disparaît."""
    torch.manual_seed(3)
    vocab_size, d = 400, 32
    echelle_embed, echelle_head = 0.02, 0.5  # deux échelles distinctes
    w_embed = torch.randn(vocab_size, d) * echelle_embed
    w_head = torch.randn(vocab_size, d) * echelle_head

    alpha_e, alpha_h = 1.0, 0.2
    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=alpha_e, alpha_h=alpha_h, lam=0.3, h=0, seed=3,
        apply_key_matrices=False,  # pour isoler le bruit de la permutation
    )

    for w_clear, w_obf, alpha in (
        (w_embed, result.w_embed_obf, alpha_e),
        (w_head, result.w_head_obf, alpha_h),
    ):
        reordonne = torch.stack([w_obf[result.permutation[t]] for t in range(vocab_size)])
        bruit = reordonne - w_clear
        attendu = alpha * float(w_clear.std())
        assert abs(float(bruit.std()) - attendu) < 0.1 * attendu, (
            f"écart-type du bruit = {float(bruit.std()):.4f}, attendu ~{attendu:.4f} "
            "(α doit être relatif à σ(W), cf. §5.2.2)"
        )


def test_embedding_obfuscation_is_a_single_consistent_linear_reparametrization():
    """Round-trip test with teeth: `result.unpermute[result.permutation[t]] ==
    t` (the brief's own test) is a tautology of any dict built as the
    inverse of another — it passes even if the permutation is applied to
    the wrong rows of w_embed_obf/w_head_obf. What actually has to hold,
    per the paper's W_tilde = Pi . W* . P_hat, is that row `permutation[t]`
    of the obfuscated matrix equals row `t` of the (noised) original matrix
    transformed by ONE fixed, invertible d x d matrix, for every token t
    simultaneously — not a per-token coincidence.

    We don't know that matrix (it's derived from a private key_matrix.py
    call the test has no access to), so we recover it by linear regression
    from a held-out subset of tokens and check it predicts the rest. A
    permutation-direction bug (Pi vs Pi^T, i.e. using `permutation` instead
    of `unpermute` for row selection) breaks this: no single matrix can
    explain a row correspondence built from mismatched pairs, so the fit
    on held-out tokens would fail to generalize. This test caught exactly
    such a bug in the brief's draft implementation.
    """
    torch.manual_seed(2)
    vocab_size, d = 60, 12  # vocab_size > d: enough rows to fit AND hold out
    w_embed = torch.randn(vocab_size, d)
    w_head = torch.randn(vocab_size, d)

    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=0.0, alpha_h=0.0, lam=0.3, h=0, seed=2
    )

    fit_tokens = list(range(d))  # d independent equations for a d x d unknown
    holdout_tokens = list(range(d, vocab_size))

    for w_clear, w_obf in (
        (w_embed, result.w_embed_obf),
        (w_head, result.w_head_obf),
    ):
        x_fit = w_clear[fit_tokens]
        y_fit = torch.stack(
            [w_obf[result.permutation[t]] for t in fit_tokens]
        )
        # x_fit is square (d, d) and generically invertible for random
        # Gaussian rows, so this recovers the transform exactly (no noise
        # here: alpha_e = alpha_h = 0.0).
        m = torch.linalg.solve(x_fit, y_fit)

        x_holdout = w_clear[holdout_tokens]
        y_holdout = torch.stack(
            [w_obf[result.permutation[t]] for t in holdout_tokens]
        )
        predicted = x_holdout @ m
        torch.testing.assert_close(predicted, y_holdout, atol=1e-4, rtol=1e-4)
