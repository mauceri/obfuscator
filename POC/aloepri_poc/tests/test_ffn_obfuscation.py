import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ffn_obfuscation import obfuscate_ffn_layer


def swiglu_ffn(x, gate_proj, up_proj, down_proj):
    gate = x @ gate_proj.T
    up = x @ up_proj.T
    hidden = torch.nn.functional.silu(gate) * up
    return hidden @ down_proj.T


def test_obfuscated_ffn_produces_identical_output():
    torch.manual_seed(0)
    hidden_size, intermediate_size = 16, 24
    gate_proj = torch.randn(intermediate_size, hidden_size)
    up_proj = torch.randn(intermediate_size, hidden_size)
    down_proj = torch.randn(hidden_size, intermediate_size)
    x = torch.randn(3, hidden_size)

    baseline_output = swiglu_ffn(x, gate_proj, up_proj, down_proj)

    obf = obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed=0)
    obf_output = swiglu_ffn(x, obf.gate_proj_obf, obf.up_proj_obf, obf.down_proj_obf)

    torch.testing.assert_close(obf_output, baseline_output, atol=1e-4, rtol=1e-4)


def test_obfuscated_ffn_weights_differ_from_original():
    torch.manual_seed(1)
    hidden_size, intermediate_size = 16, 24
    gate_proj = torch.randn(intermediate_size, hidden_size)
    up_proj = torch.randn(intermediate_size, hidden_size)
    down_proj = torch.randn(hidden_size, intermediate_size)

    obf = obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed=1)
    assert not torch.allclose(obf.gate_proj_obf, gate_proj)
    assert not torch.allclose(obf.up_proj_obf, up_proj)
