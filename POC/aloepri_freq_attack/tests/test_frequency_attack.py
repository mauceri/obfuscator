import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from frequency_attack import (
    build_frequency_ranking,
    random_permutation,
    apply_permutation,
    tfma_recovery_rate,
)


def test_build_frequency_ranking_orders_by_descending_count():
    tokens = [1, 1, 1, 2, 2, 3]
    assert build_frequency_ranking(tokens) == [1, 2, 3]


def test_permutation_round_trip():
    vocab = list(range(20))
    perm = random_permutation(vocab, seed=0)
    inverse = {v: k for k, v in perm.items()}
    tokens = [3, 7, 19, 0]
    permuted = apply_permutation(tokens, perm)
    restored = apply_permutation(permuted, inverse)
    assert restored == tokens


def test_tfma_recovers_dominant_token_with_enough_data():
    vocab = [0, 1, 2, 3, 4]
    weights = [50, 20, 15, 10, 5]
    permutation = random_permutation(vocab, seed=1)

    rng = random.Random(42)
    reference_tokens = rng.choices(vocab, weights=weights, k=20000)
    observed_clear = rng.choices(vocab, weights=weights, k=20000)
    observed_permuted = apply_permutation(observed_clear, permutation)

    rate = tfma_recovery_rate(observed_permuted, reference_tokens, permutation, top_k=1)
    assert rate == 1.0


def test_tfma_recovery_rate_is_bounded():
    vocab = list(range(10))
    weights = [1] * 10
    permutation = random_permutation(vocab, seed=2)
    rng = random.Random(7)
    reference_tokens = rng.choices(vocab, weights=weights, k=5000)
    observed_permuted = apply_permutation(
        rng.choices(vocab, weights=weights, k=5000), permutation
    )
    rate = tfma_recovery_rate(observed_permuted, reference_tokens, permutation, top_k=5)
    assert 0.0 <= rate <= 1.0
