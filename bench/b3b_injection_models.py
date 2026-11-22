"""B3b -- the injection model dominates every budget once tool results are
scanned. Options, measured at 120 / 482 tokens (short prompt / 2 KB tool result):

  base_torch_fp32     deberta-v3-base, PyTorch fp32            (current README setup)
  base_torch_int8dyn  same, torch dynamic int8 on nn.Linear
  base_onnx_fp32      same weights exported to ONNX (fidelity checked below)
  base_onnx_int8      ONNX + onnxruntime dynamic int8
  xsmall_torch_fp32   deberta-v3-xsmall config (Llama Prompt Guard 2 22M class)
  xsmall_onnx_int8    xsmall, ONNX int8

Export fidelity: the README kept DeBERTa on PyTorch because an ONNX export
"scores wrong". With identical (random) weights, fp32 ONNX logits must match
PyTorch logits; we report max |diff| over varied inputs. That checks the
*graph*, independent of trained weights. int8 variants change numerics by
design: their ACCURACY must be validated on the real weights (evals/run_eval.py)
before use -- this script only measures speed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

import common

BASE = dict(vocab_size=128100, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            intermediate_size=3072)
XSMALL = dict(vocab_size=128100, hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
              intermediate_size=1536)
COMMON = dict(hidden_act="gelu", max_position_embeddings=512, type_vocab_size=0,
              relative_attention=True, position_buckets=256, norm_rel_ebd="layer_norm",
              share_att_key=True, pos_att_type=["p2c", "c2p"], max_relative_positions=-1,
              position_biased_input=False, layer_norm_eps=1e-7, num_labels=2, pad_token_id=0)


def build(cfg):
    torch.manual_seed(0)
    return DebertaV2ForSequenceClassification(DebertaV2Config(**cfg, **COMMON)).eval()


def export(model, path):
    ids = torch.randint(5, 1000, (1, 32))
    torch.onnx.export(model, (ids, torch.ones_like(ids)), path,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}}, opset_version=17, dynamo=False)


def quant(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def fidelity(model, sess, lengths=(7, 33, 120, 257, 482, 512)):
    worst = 0.0
    rng = np.random.default_rng(0)
    for n in lengths:
        ids = rng.integers(5, 120000, size=(1, n)).astype(np.int64)
        with torch.inference_mode():
            a = model(input_ids=torch.from_numpy(ids), attention_mask=torch.ones(1, n, dtype=torch.long)).logits.numpy()
        b = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


def bench_torch(model, n, rep):
    ids = torch.randint(5, 120000, (1, n))
    m = torch.ones_like(ids)

    def f():
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=m)
    return common.timeit_ms(f, repeat=rep, warmup=2)


def bench_onnx(sess, n, rep):
    ids = np.random.default_rng(1).integers(5, 120000, size=(1, n)).astype(np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    return common.timeit_ms(lambda: sess.run(None, feeds), repeat=rep, warmup=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=8)
    args = ap.parse_args()
    tmp = tempfile.mkdtemp()
    out = {"fidelity_max_abs_logit_diff": {}, "rows": []}
    for size, cfg in (("base", BASE), ("xsmall", XSMALL)):
        model = build(cfg)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        fp32, int8 = f"{tmp}/{size}.onnx", f"{tmp}/{size}.int8.onnx"
        export(model, fp32)
        quant(fp32, int8)
        s32 = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
        out["fidelity_max_abs_logit_diff"][f"{size}_onnx_fp32_vs_torch"] = fidelity(model, s32)
        print(size, f"{params:.0f}M params; ONNX fp32 vs torch max|dlogit| =",
              out["fidelity_max_abs_logit_diff"][f"{size}_onnx_fp32_vs_torch"], flush=True)
        variants = {f"{size}_torch_fp32": ("torch", model), f"{size}_onnx_int8": ("onnx", s8)}
        if size == "base":
            variants["base_torch_int8dyn"] = ("torch", torch.ao.quantization.quantize_dynamic(
                build(cfg), {torch.nn.Linear}, dtype=torch.qint8))
            variants["base_onnx_fp32"] = ("onnx", s32)
        for n in (120, 482):
            for name, (kind, obj) in variants.items():
                s = bench_torch(obj, n, args.rep) if kind == "torch" else bench_onnx(obj, n, args.rep)
                out["rows"].append({"variant": name, "params_M": round(params, 1), "tokens": n, **s})
                print(f"{name:20s} {n:4d} tok  p50={s['p50_ms']:8.1f} p95={s['p95_ms']:8.1f}", flush=True)
    common.write("b3b_injection_models", out)


if __name__ == "__main__":
    main()
