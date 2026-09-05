"""Simulation d'attaque par fréquence de tokens (TFMA-style)."""
from collections import Counter
import random


def build_frequency_ranking(token_ids):
    """Rang de fréquence décroissante : ranking[0] = token le plus fréquent."""
    counts = Counter(token_ids)
    return [tok for tok, _ in counts.most_common()]


def random_permutation(vocab_ids, seed):
    rng = random.Random(seed)
    shuffled = list(vocab_ids)
    rng.shuffle(shuffled)
    return dict(zip(vocab_ids, shuffled))


def apply_permutation(token_ids, permutation):
    return [permutation[t] for t in token_ids]


def tfma_recovery_rate(observed_permuted_ids, reference_token_ids, permutation, top_k):
    """
    Simule TFMA : l'attaquant classe les IDs permutés observés par fréquence,
    classe un corpus de référence en clair par fréquence, et suppose que le
    rang k du classement observé correspond au rang k du classement de
    référence. Mesure le % de tokens des top_k les plus fréquents (en clair)
    correctement retrouvés.
    """
    if top_k <= 0:
        return 0.0

    observed_ranking = build_frequency_ranking(observed_permuted_ids)
    reference_ranking = build_frequency_ranking(reference_token_ids)
    inverse_permutation = {v: k for k, v in permutation.items()}

    correct = 0
    for rank in range(top_k):
        if rank >= len(observed_ranking) or rank >= len(reference_ranking):
            break
        guessed_clear_tok = reference_ranking[rank]
        true_clear_tok = inverse_permutation.get(observed_ranking[rank])
        if guessed_clear_tok == true_clear_tok:
            correct += 1
    return correct / top_k
