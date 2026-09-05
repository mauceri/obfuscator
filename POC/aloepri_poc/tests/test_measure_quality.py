"""Tests de la logique pure de `measure_quality.py` — tout ce qui ne demande
ni GPU ni vrai modèle. Le calcul de perplexité lui-même (`perplexity()`)
n'est pas testé ici : il a besoin d'un vrai modèle chargé, cf. RUNBOOK.md."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from measure_quality import TEST_PROMPTS, permute_ids


def test_permute_ids_applies_the_permutation():
    permutation = {1: 5, 2: 6, 3: 7}
    assert permute_ids([1, 2, 3], permutation) == [5, 6, 7]


def test_permute_ids_is_a_noop_without_a_permutation():
    """C'est le chemin baseline : `perplexity(..., permutation=None)`."""
    assert permute_ids([1, 2, 3], None) == [1, 2, 3]


def test_permute_ids_does_not_mutate_its_input():
    ids = [1, 2, 3]
    permute_ids(ids, None)
    assert ids == [1, 2, 3]


def test_test_prompts_is_a_representative_fixed_set():
    """Sanity sur le jeu de prompts fixe : le brouillon du plan demandait
    ~20-30 prompts factuels courts, pas 3 exemples de démonstration."""
    assert 20 <= len(TEST_PROMPTS) <= 30
    assert len(set(TEST_PROMPTS)) == len(TEST_PROMPTS)  # pas de doublon
    assert all(isinstance(p, str) and p.strip() for p in TEST_PROMPTS)
