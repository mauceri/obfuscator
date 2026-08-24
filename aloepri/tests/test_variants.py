"""Les variantes de l'arc d'attaques : désactivation sélective des briques.

`obfuscate_attention=False` : les poids d'attention restent byte-identiques
(le FFN reste obfusqué).
`apply_permutation=False` : l'embedding/lm_head gardent l'ordre clair des
lignes (le bruit α est conservé), la permutation retournée est l'identité et
les token_ids spéciaux ne sont pas remappés — le canal hidden se mesure dans
l'espace clair.
"""
import torch

from ..model_transform import obfuscate_model_in_place
from .test_qwen3_arch import _tiny_qwen3  # réutilise la miniature Qwen3


def _roundtrip_ok(model, ids, perm):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1]
    # sans permutation, l'id du top-1 clair doit rester lisible
    return int(logits.argmax())  # id clair attendu


def test_variant_no_attention_keeps_attention_weights():
    """`obfuscate_attention=False` : les poids d'attention (q/k/v/o) restent
    byte-identiques — seule l'attention est désactivée, pas le FFN."""
    model, config = _tiny_qwen3()
    layer0 = model.model.layers[0]
    attn0 = {n: p.weight.data.clone()
             for n, p in layer0.self_attn.named_children()
             if n.endswith("_proj")}
    gate0 = layer0.mlp.gate_proj.weight.data.clone()

    obfuscate_model_in_place(model, config, seed=0, obfuscate_attention=False)

    for name, before in attn0.items():
        after = dict(layer0.self_attn.named_children())[name]
        assert torch.equal(after.weight.data, before), (
            f"attention doit rester intacte ({name})")
    # le FFN, lui, est toujours obfusqué
    assert not torch.allclose(layer0.mlp.gate_proj.weight.data, gate0), (
        "le FFN doit rester obfusqué")


def test_variant_no_permutation_keeps_clear_ids():
    """`apply_permutation=False` : la table d'embedding garde l'ordre clair —
    la permutation retournée est l'identité et l'espace de sortie reste un
    vocabulaire d'ids clairs (le canal hidden se mesure dans l'espace clair)."""
    model, config = _tiny_qwen3()
    ids = [1, 2, 3]
    keys = obfuscate_model_in_place(model, config, seed=0,
                                    apply_permutation=False)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits
    assert logits.shape[-1] == config.vocab_size
    # sans permutation, l'id du top-1 est directement un id clair
    top1 = _roundtrip_ok(model, ids, keys.vocab_permutation)
    assert 0 <= top1 < config.vocab_size
    # la permutation retournée est l'identité : l'espace clair est préservé
    for i in range(config.vocab_size):
        assert keys.vocab_permutation[i] == i
        assert keys.vocab_unpermute[i] == i
