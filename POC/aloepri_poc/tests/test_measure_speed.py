"""Tests de la logique pure de `measure_speed.py`. `measure()` lui-même n'est
pas testé ici : il appelle `model.generate()` sur un vrai modèle/GPU, cf.
RUNBOOK.md."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from measure_speed import tokens_per_second


def test_tokens_per_second_basic_ratio():
    assert tokens_per_second(100, 10.0) == 10.0


def test_tokens_per_second_with_fractional_elapsed():
    assert tokens_per_second(50, 2.5) == 20.0
