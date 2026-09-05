"""
Mesure réelle de l'Étape 0 : volume de corpus nécessaire pour qu'un
attaquant retrouve les tokens les plus fréquents via TFMA, sur le vrai
tokenizer Qwen2.5-7B et un corpus français réel.

Usage: python run_frequency_experiment.py
Nécessite : `pip install transformers datasets`, accès réseau (télécharge
le tokenizer et un échantillon Wikipedia FR en streaming).
"""
from transformers import AutoTokenizer
from datasets import load_dataset

from frequency_attack import random_permutation, apply_permutation, tfma_recovery_rate


def load_token_stream(tokenizer, n_articles):
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    tokens = []
    for i, row in enumerate(ds):
        if i >= n_articles:
            break
        tokens.extend(tokenizer.encode(row["text"]))
    return tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    vocab_ids = list(range(tokenizer.vocab_size))
    permutation = random_permutation(vocab_ids, seed=0)

    all_tokens = load_token_stream(tokenizer, n_articles=200)
    reference_tokens = all_tokens[: len(all_tokens) // 2]
    target_tokens = all_tokens[len(all_tokens) // 2 :]

    print("volume_observé\ttop10\ttop100\ttop1000")
    for n in [100, 1000, 10000, 100000, len(target_tokens)]:
        n = min(n, len(target_tokens))
        observed_clear = target_tokens[:n]
        observed_permuted = apply_permutation(observed_clear, permutation)
        rates = [
            tfma_recovery_rate(observed_permuted, reference_tokens, permutation, k)
            for k in (10, 100, 1000)
        ]
        print(f"{n}\t{rates[0]:.3f}\t{rates[1]:.3f}\t{rates[2]:.3f}")


if __name__ == "__main__":
    main()
