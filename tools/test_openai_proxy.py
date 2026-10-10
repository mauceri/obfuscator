"""Tests du proxy : faux tokenizer, permutation identité, faux Modal."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import openai_proxy  # noqa: E402


class Identite:
    def __getitem__(self, i):
        return i


class FauxCodec:
    permutation = unpermute = Identite()


class FauxTokenizer:
    def __init__(self, sortie):
        self.sortie = sortie
        self.messages = None

    def apply_chat_template(self, messages, tools=None, **kw):
        self.messages = messages
        return "gabarit"

    def __call__(self, texte):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids, skip_special_tokens=True):
        return self.sortie


@pytest.fixture
def proxy(monkeypatch):
    envois = []

    class Rep:
        status_code = 200

        def json(self):
            return {"output_ids": envois[-1]["input_ids"] + [100, 101]}

    def post(url, json=None, **kw):
        envois.append(json)
        return Rep()
    monkeypatch.setattr(openai_proxy.requests, "post", post)

    def faire(sortie):
        tok = FauxTokenizer(sortie)
        app = openai_proxy.make_app("http://modal", "", FauxCodec(), tok, "m")
        return TestClient(app), tok, envois
    return faire


def _evenements(rep):
    lignes = [l[len("data: "):] for l in rep.text.splitlines() if l.startswith("data: ")]
    assert lignes[-1] == "[DONE]"
    return [json.loads(l) for l in lignes[:-1]]


def test_stream_texte(proxy):
    client, _, _ = proxy("Bonjour")
    rep = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "stream": True,
        "stream_options": {"include_usage": True}})
    assert rep.status_code == 200 and rep.headers["content-type"].startswith("text/event-stream")
    ev = _evenements(rep)
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in ev if e["choices"]) == "Bonjour"
    assert [e["choices"][0]["finish_reason"] for e in ev if e["choices"]][-1] == "stop"
    assert ev[-1]["usage"]["completion_tokens"] == 2


def test_stream_appel_d_outil(proxy):
    client, _, _ = proxy('<tool_call>{"name": "wiki_review", "arguments": {}}</tool_call>')
    rep = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "stream": True})
    ev = _evenements(rep)
    appels = [tc for e in ev for tc in e["choices"][0]["delta"].get("tool_calls") or []]
    assert appels[0]["index"] == 0 and appels[0]["function"]["name"] == "wiki_review"
    assert ev[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_max_completion_tokens_prioritaire(proxy):
    client, _, envois = proxy("ok")
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "max_tokens": 10,
        "max_completion_tokens": 5000})
    assert envois[-1]["max_new_tokens"] == 2048


def test_historique_des_outils_transmis(proxy):
    client, tok, _ = proxy("ok")
    appel = {"id": "call_1", "type": "function",
             "function": {"name": "wiki_review", "arguments": "{}"}}
    client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [appel]},
        {"role": "tool", "tool_call_id": "call_1", "content": "page"}]})
    assert tok.messages[1]["tool_calls"] == [appel]
    assert tok.messages[2]["tool_call_id"] == "call_1"


def test_sans_stream_inchange(proxy):
    client, _, _ = proxy("Bonjour")
    rep = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert rep.json()["choices"][0]["message"]["content"] == "Bonjour"
