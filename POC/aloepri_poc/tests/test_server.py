"""Tests de `server.py` qui ne demandent ni GPU ni vrai modèle : le schéma de
requête/réponse HTTP et le câblage `/generate`, avec un modèle factice injecté
à la place du vrai `AutoModelForCausalLM`.

Ce que ces tests NE couvrent PAS, volontairement : le chargement du vrai
modèle 7B obfusqué (`load()`), qui demande un GPU — cf. RUNBOOK.md pour la
vérification manuelle sur le Pod (round-trip de bout en bout, Step 4)."""
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

pytest.importorskip("fastapi")
import torch
from fastapi.testclient import TestClient

import server


class FakeModel:
    """Modèle factice : renvoie l'entrée suivie d'IDs incrémentaux, sans
    aucune dépendance à un vrai checkpoint ni à un GPU."""
    device = "cpu"

    def generate(self, input_tensor, max_new_tokens, do_sample):
        assert do_sample is False  # génération déterministe attendue (greedy)
        extra = list(range(max_new_tokens))
        return torch.tensor([input_tensor[0].tolist() + extra])


@pytest.fixture
def client():
    server._model = FakeModel()
    yield TestClient(server.app)
    server._model = None


def test_generate_returns_input_followed_by_generated_ids(client):
    resp = client.post("/generate", json={"input_ids": [10, 20, 30], "max_new_tokens": 5})
    assert resp.status_code == 200
    assert resp.json() == {"output_ids": [10, 20, 30, 0, 1, 2, 3, 4]}


def test_generate_uses_default_max_new_tokens_of_100(client):
    resp = client.post("/generate", json={"input_ids": [1]})
    assert resp.status_code == 200
    assert len(resp.json()["output_ids"]) == 1 + 100


def test_generate_rejects_missing_input_ids(client):
    resp = client.post("/generate", json={"max_new_tokens": 5})
    assert resp.status_code == 422


def test_generate_rejects_non_integer_input_ids(client):
    resp = client.post("/generate", json={"input_ids": ["a", "b"]})
    assert resp.status_code == 422


def test_server_module_does_not_import_autotokenizer():
    """Garde-fou de la Task 9 (contrainte obligatoire n°1) : `server.py` ne
    doit jamais importer/appeler `AutoTokenizer.from_pretrained` — le
    répertoire du modèle obfusqué ne contient aucun fichier de tokenizer, cet
    appel planterait à l'exécution réelle sur le Pod. On vérifie l'import
    réel (pas le texte de la docstring, qui explique justement pourquoi)."""
    assert not hasattr(server, "AutoTokenizer")
    assert "from transformers import AutoModelForCausalLM\n" in (BASE / "server.py").read_text()
