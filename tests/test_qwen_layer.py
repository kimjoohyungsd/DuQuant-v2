# coding=utf-8
"""Smoke test for models/int_qwen_layer.py.

    python tests/test_qwen_layer.py

Builds a tiny Qwen2 model (available in every transformers that ships Qwen) and a
tiny Qwen3-shaped variant (q_norm/k_norm bolted on, so the Qwen3-specific code path
is exercised even on installs without transformers.models.qwen3), wraps each layer
in QuantQwenDecoderLayer, and checks:

  1. at 16 bits with the rotation disabled the wrapper is a no-op -- outputs must
     match the original layer, which is what proves the attention rewrite is faithful
     (q_norm/k_norm placement, head_dim handling, RoPE, GQA repeat);
  2. at W4A4 MXFP4 with DuQuant rotation the layer runs and stays finite.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm  # noqa: E402

from models.int_qwen_layer import QuantQwenDecoderLayer  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def make_args(bits, quant_method):
    """Mimic the dicts main.py builds."""
    w = dict(n_bits=bits, per_channel_axes=[0], symmetric=False,
             dynamic_method="per_channel", group_size=None, lwc=False, swc=None,
             quant_method=quant_method, block_size=32, max_rotation_step=8,
             permutation_times=0)
    a = dict(n_bits=bits, per_channel_axes=[], symmetric=False, lac=None,
             act_group_size=None, dynamic_method="per_token",
             quant_method=quant_method, block_size=32, max_rotation_step=8,
             permutation_times=0)
    args = types.SimpleNamespace()
    for k in ("q", "k", "v", "o", "gate", "up", "down"):
        setattr(args, f"{k}_weight_quant_params", dict(w))
        setattr(args, f"{k}_act_quant_params", dict(a))
    return args


def tiny_layer(qwen3_style):
    cfg = transformers.Qwen2Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=64, rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    layer = Qwen2DecoderLayer(cfg, layer_idx=0).eval()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    if qwen3_style:
        # Qwen3 = Qwen2 + per-head q/k RMSNorm, and no qkv bias.
        layer.self_attn.q_norm = Qwen2RMSNorm(head_dim, eps=cfg.rms_norm_eps)
        layer.self_attn.k_norm = Qwen2RMSNorm(head_dim, eps=cfg.rms_norm_eps)
        with torch.no_grad():
            layer.self_attn.q_norm.weight.normal_(1.0, 0.1)
            layer.self_attn.k_norm.weight.normal_(1.0, 0.1)
    return cfg, layer


def reference_forward(layer, x, mask, pos_ids, qwen3_style):
    """Original Qwen2DecoderLayer, with q_norm/k_norm inserted the way Qwen3 does."""
    if not qwen3_style:
        return layer(x, attention_mask=mask, position_ids=pos_ids)[0]
    # Qwen2DecoderLayer has no q_norm hook, so emulate the Qwen3 block by hand.
    import math
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    a = layer.self_attn
    b, t, _ = x.shape
    hd = a.head_dim
    residual = x
    h = layer.input_layernorm(x)
    q = a.q_proj(h).view(b, t, a.num_heads, hd).transpose(1, 2)
    k = a.k_proj(h).view(b, t, a.num_key_value_heads, hd).transpose(1, 2)
    v = a.v_proj(h).view(b, t, a.num_key_value_heads, hd).transpose(1, 2)
    q, k = a.q_norm(q), a.k_norm(k)
    import inspect
    if 'seq_len' in inspect.signature(a.rotary_emb.forward).parameters:
        cos, sin = a.rotary_emb(v, seq_len=t)
    else:
        cos, sin = a.rotary_emb(v, pos_ids)
    _p = inspect.signature(apply_rotary_pos_emb).parameters
    if 'unsqueeze_dim' in _p and 'position_ids' not in _p:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    else:
        q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
    k = repeat_kv(k, a.num_key_value_groups)
    v = repeat_kv(v, a.num_key_value_groups)
    w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(hd)
    if mask is not None:
        w = w + mask[:, :, :, : k.shape[-2]]
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(w, v).transpose(1, 2).reshape(b, t, a.num_heads * hd)
    h = residual + a.o_proj(o)
    residual = h
    h2 = layer.post_attention_layernorm(h)
    return residual + layer.mlp(h2)


@torch.no_grad()
def test_identity(qwen3_style, device):
    # fp16: QuantQwenDecoderLayer.forward hard-codes .half() before the MLP, exactly
    # as the llama layer it mirrors does, so the wrapper is fp16-only by construction.
    tag = "Qwen3-style (q_norm/k_norm)" if qwen3_style else "Qwen2-style"
    cfg, layer = tiny_layer(qwen3_style)
    layer = layer.half().to(device)
    x = torch.randn(1, 12, cfg.hidden_size, dtype=torch.float16, device=device)
    pos_ids = torch.arange(12, device=device).unsqueeze(0)
    mask = torch.zeros(1, 1, 12, 12, dtype=torch.float16, device=device)
    mask = mask.masked_fill(
        torch.triu(torch.ones(12, 12, device=device), 1).bool(),
        torch.finfo(torch.float16).min)

    ref = reference_forward(layer, x, mask, pos_ids, qwen3_style)

    q = QuantQwenDecoderLayer(cfg, layer, make_args(16, None), layer_idx=0)
    q = q.half().to(device).eval()
    q.let = False
    got = q(x, attention_mask=mask, position_ids=pos_ids)[0]

    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    check(f"16-bit wrapper is a no-op vs reference ({tag})",
          err < 5e-3 * max(scale, 1.0),
          f"max abs diff = {err:.2e} (output scale {scale:.2e})")


@torch.no_grad()
def test_quantized_runs(qwen3_style):
    tag = "Qwen3-style" if qwen3_style else "Qwen2-style"
    cfg, layer = tiny_layer(qwen3_style)
    layer = layer.half().cuda()
    x = torch.randn(1, 12, cfg.hidden_size, dtype=torch.float16, device="cuda")
    pos_ids = torch.arange(12, device="cuda").unsqueeze(0)

    q = QuantQwenDecoderLayer(cfg, layer, make_args(4, "duquant"), layer_idx=0)
    q = q.half().cuda().eval()
    q.let = False
    q.set_quant_state(weight_quant=True, act_quant=True)
    out = q(x, position_ids=pos_ids)[0]
    check(f"W4A4 MXFP4 + DuQuant rotation runs and is finite ({tag})",
          torch.isfinite(out).all().item() and out.shape == x.shape,
          f"out {tuple(out.shape)}")
    # the helper methods duquant.py calls must exist
    for m in ("smooth_and_quant_temporary", "smooth_and_quant_inplace",
              "register_duquant_params", "register_scales_and_zeros",
              "duquant_state_dict", "load_duquant_params", "clear_temp_variable"):
        if not hasattr(q, m):
            check(f"decoder layer exposes {m}", False)
            return
    check(f"decoder layer exposes every method duquant.py calls ({tag})", True)


def main():
    print(f"Qwen layer smoke test (transformers {transformers.__version__})\n")
    if not torch.cuda.is_available():
        print("needs CUDA (the layer hard-codes .half())")
        return 1
    print("[1] faithfulness at 16 bits, rotation off")
    test_identity(False, "cuda")
    test_identity(True, "cuda")
    if True:
        print("[2] quantized path")
        test_quantized_runs(False)
        test_quantized_runs(True)
    print("\nRESULT:", "PASS" if not FAILED else f"FAIL ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
