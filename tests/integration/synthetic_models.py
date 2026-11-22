"""Build SYNTHETIC models in the exact ./models layout that model-init produces,
so the real guardrails service can be exercised without network or torch.

They are keyword-triggered (embedding lookup + max over tokens): their scores
are meaningless as a measure of model quality. They exist to exercise every
code path between a request and a decision -- tokenizer, windowing, ONNX
Runtime, backend auto-detection, manifest verification, /v1/decide, the
multi-label (OpenJev slot) backend and its OWASP mapping."""
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "guardrails"))
import manifest  # noqa: E402

TRIGGERS = {
    "injection": ["ignore", "disregard", "override", "reveal", "jailbreak", "dan", "unrestricted",
                  "instructions", "bypass"],
    "toxicity": ["worthless", "idiot", "despise", "despises", "pathetic", "hate", "stupid", "moron",
                 "disgusting"],
    "system_prompt_extraction": ["prompt", "hidden", "reveal"],
}


def _vocab():
    words = {"[PAD]", "[UNK]"}
    for fn in os.listdir(os.path.join(ROOT, "evals", "datasets")):
        for line in open(os.path.join(ROOT, "evals", "datasets", fn)):
            words.update(re.findall(r"[a-z0-9]+", line.lower()))
    for ws in TRIGGERS.values():
        words.update(ws)
    words.update("role user tool assistant fetch search".split())
    return ["[PAD]", "[UNK]"] + sorted(words - {"[PAD]", "[UNK]"})


def _tokenizer(path, vocab):
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    # Regex(...) -- a plain str pattern is matched LITERALLY by tokenizers' Split
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[^a-z0-9]+"), "removed")
    PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                            model_max_length=512).save_pretrained(path)


def _onnx(path, vocab, label_triggers, safe_column=False):
    """logits[:, j] = max over tokens of E[token, j]; trigger words fire their
    column. safe_column: column 0 is a constant 0 logit (binary classifier with
    softmax: p(label) ~ 0.007 when nothing fires, ~0.993 when a trigger does)."""
    n = len(label_triggers)
    E = np.full((len(vocab), n), -5.0, dtype=np.float32)
    for j, trig in enumerate(label_triggers):
        for w in trig:
            if w in vocab:
                E[vocab.index(w), j] = 5.0
    if safe_column:
        E[:, 0] = 0.0
    E[0] = -20.0
    g = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["e"]),
         helper.make_node("ReduceMax", ["e"], ["logits"], axes=[1], keepdims=0)],
        "synthetic", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "T"]),
                      helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", n])],
        [numpy_helper.from_array(E, "E")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def build_stock(root):
    """./models/{injection,toxicity} + MANIFEST.sha256, like model-init."""
    vocab = _vocab()
    for name in ("injection", "toxicity"):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        _tokenizer(d, vocab)
        # single-label, POS_LABEL=1: column 0 = "safe" (never fires), column 1 = label
        _onnx(os.path.join(d, "model_quantized.onnx"), vocab, [[], TRIGGERS[name]], safe_column=True)
        json.dump({"synthetic": True}, open(os.path.join(d, "source.json"), "w"))
    os.makedirs(os.path.join(root, "decision"), exist_ok=True)
    manifest.write(root, ["injection", "toxicity"])


def build_decision(root, name="openjev-slot-conformance"):
    """./models/decision: a multi-label System-1 model in the OpenJev slot
    format, declaring its own OWASP mapping, with its own manifest."""
    vocab = _vocab()
    d = os.path.join(root, "decision")
    os.makedirs(d, exist_ok=True)
    labels = ["injection", "toxicity", "system_prompt_extraction"]
    _tokenizer(d, vocab)
    _onnx(os.path.join(d, "model_quantized.onnx"), vocab, [TRIGGERS[l] for l in labels])
    json.dump({"name": name, "version": "0-synthetic", "labels": labels, "activation": "sigmoid",
               "owasp": {"injection": ["LLM01:2025"], "toxicity": [],
                         "system_prompt_extraction": ["LLM07:2025", "LLM01:2025"]}},
              open(os.path.join(d, "labels.json"), "w"), indent=2)
    manifest.write(d, ["."])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models")
    build_stock(out)
    if "--decision" in sys.argv:
        build_decision(out)
    print("synthetic models written to", out)
